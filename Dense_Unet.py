import os
import pandas as pd

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch import optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
from torchvision import transforms

class DenseUNet(nn.Module):
    def __init__(self, num_classes, weights=None):
        super(DenseUNet, self).__init__()

        if weights:
            densenet = models.densenet121()
        else:
            densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        self.features = densenet.features
        self.features.conv0.stride = 1

        self.up3 = UpSampling(1024, 512)  # 입력 채널과 출력 채널 설정
        self.up2 = UpSampling(512, 256)
        self.up1 = UpSampling(256, 128)
        self.up0 = UpSampling(128, 64)

        self.res = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)

        x = self.features.denseblock1(x)
        skip1 = x
        x = self.features.transition1(x)

        x = self.features.denseblock2(x)
        skip2 = x
        x = self.features.transition2(x)

        x = self.features.denseblock3(x)
        skip3 = x
        x = self.features.transition3(x)

        x = self.features.denseblock4(x)

        x = self.up3(x, skip3)

        x = self.up2(x, skip2)

        x = self.up1(x, skip1)

        x = self.res(x)

        return x


class UpSampling(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UpSampling, self).__init__()

        self.sequence1 = nn.ConvTranspose2d(input_channels, output_channels * 2, kernel_size=2, stride=2)

        self.sequence2 = nn.Sequential(
            nn.BatchNorm2d(output_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels * 4, output_channels, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x, skip_connection):
        x = self.sequence1(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.sequence2(x)

        return x

class MultiClassDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(MultiClassDiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, preds, targets):
        # preds와 targets는 (Batch, Classes, Height, Width)의 형태여야 합니다.
        preds = torch.softmax(preds, dim=1)  # 예측값을 확률로 변환
        num_classes = preds.shape[1]

        total_dice = 0
        for class_index in range(num_classes):
            preds_flat = preds[:, class_index, :, :].contiguous().view(-1)
            targets_flat = (targets == class_index).contiguous().view(-1).float()

            intersection = (preds_flat * targets_flat).sum()
            dice = (2. * intersection + self.epsilon) / (preds_flat.sum() + targets_flat.sum() + self.epsilon)
            total_dice += dice

        mean_dice = total_dice / num_classes
        return 1 - mean_dice


def mIoU(pred, label, num_classes):
    ious = []
    for class_id in range(num_classes):
        # 실제와 예측에서 해당 클래스에 해당하는 마스크 생성
        true_class = (label == class_id).int()
        pred_class = (pred == class_id).int()

        # 교집합 및 합집합 계산
        intersection = torch.sum(true_class & pred_class).float()
        union = torch.sum(true_class | pred_class).float()

        if union == 0:
            ious.append(-1)
        else:
            iou = intersection / union
            ious.append(iou)

    sum_iou = 0.0
    cnt = 0

    for iou in ious:
        if iou != -1:
            sum_iou += iou
            cnt += 1

    if cnt == 0:
        return 0.0

    miou = sum_iou/cnt

    return miou

# 학습 루프
def train_model(model, train_loader, epochs=50, learning_rate=0.001):
    criterion = MultiClassDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    scaler = GradScaler(init_scale=1024, growth_factor=2.0, enabled=True)

    loss_history = []
    iou_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_iou = 0.0

        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device).long()
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                iou = mIoU(outputs.argmax(dim=1), labels, 30)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_iou += iou.item()

            # 통계 출력
            print(f'\r[{epoch + 1}, {i + 1}] loss: {loss.item():.3f}, IoU: {iou.item():.3f}', end='')

        print()


        epoch_loss /= len(train_loader)
        epoch_iou /= len(train_loader)

        scheduler.step(epoch_iou)

        print(f'\r[{epoch + 1}] loss: {epoch_loss:.3f}, IoU: {epoch_iou:.3f}', end='')

        print()
        print()

        loss_history.append(epoch_loss)
        iou_history.append(epoch_iou)

        torch.save(model.state_dict(), save_dir + "/model.pth")

        # 데이터프레임으로 변환
        history_df = pd.DataFrame({
            'Epoch': range(1, len(loss_history) + 1),
            'Loss': loss_history,
            'IoU': iou_history
        })

        # CSV 파일로 저장
        history_df.to_csv(save_dir + '/training_history.csv', index=False)

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        # 이미지 파일명 리스트
        image_filenames = sorted(os.listdir(image_dir))

        # 라벨 파일명 리스트
        label_filenames = sorted(os.listdir(label_dir))

        # 이미지 경로와 라벨 경로 리스트
        self.image_paths = [image_dir + '/' + fname for fname in image_filenames]
        self.label_paths = [label_dir + '/' + fname for fname in label_filenames]

        print(self.image_paths)
        print(self.label_paths)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 이미지 파일 경로
        image_path = self.image_paths[idx]

        # 라벨 파일 경로 (visualized_ 접두어 추가)
        label_path = self.label_paths[idx]

        # 이미지와 라벨 불러오기
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        label_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        ])

        image = transform(image)
        label = label_transform(label)
        label = torch.from_numpy(np.array(label)).long()

        return image, label

def predict(model, test_dataset, device):
    model.eval()  # 모델을 평가 모드로 설정

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    original_dataset = test_dataset.dataset  # Subset에서 원본 CustomDataset에 접근

    for i in range(len(test_dataset)):
        image, _ = original_dataset[test_dataset.indices[i]]  # Subset의 인덱스를 사용하여 원본 데이터셋에서 이미지를 가져옴

        image_path = original_dataset.image_paths[test_dataset.indices[i]]
        image_name = os.path.basename(image_path)

        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            with autocast(device_type='cuda'):
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)

        res = predicted.squeeze(0).cpu().numpy()

        color_image = to_color(res, image_name)

        res = Image.fromarray(res.astype(np.uint8))  # 0~255 범위로 변환하여 저장
        output_path = os.path.join(save_dir, f'predicted/predicted_{image_name}')
        res.save(output_path, format="PNG")

def to_color(image, image_name):
    colors = {
        0: [128, 64, 128], 1: [244, 35, 232], 2: [250, 170, 160], 3: [230, 150, 140],
        4: [220, 20, 60], 5: [255, 0, 0],
        6: [0, 0, 142], 7: [0, 0, 70], 8: [0, 60, 100], 9: [0, 80, 100], 10: [0, 0, 230],
        11: [119, 11, 32], 12: [0, 0, 90], 13: [0, 0, 110],
        14: [70, 70, 70], 15: [102, 102, 156], 16: [190, 153, 153], 17: [180, 165, 180],
        18: [150, 100, 100], 19: [150, 120, 90],
        20: [153, 153, 153], 21: [153, 153, 153], 22: [220, 220, 0],
        23: [250, 170, 30],
        24: [107, 142, 35], 25: [152, 251, 152],
        26: [70, 130, 180],
        27: [81, 0, 81], 28: [111, 74, 0], 29: [81, 0, 21]
    }

    # 컬러 이미지 초기화 (높이, 너비, 3채널)
    color_image = np.zeros((*image.shape, 3), dtype=np.uint8)

    # 각 픽셀 값을 색상 매핑
    for gray_value, rgb in colors.items():
        color_image[image == gray_value] = rgb

    # NumPy 배열을 PIL 이미지로 변환
    res = Image.fromarray(color_image)
    res.save(os.path.join(save_dir, f'predicted_color/predicted_color_{image_name}'))

    return color_image

data_dir = "C:/picture/ip_dataset/image"
label_dir = "C:/picture/ip_dataset/label"
save_dir = "C:/picture/ip_dataset/DENSE_UNET_MODEL"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 1
EPOCH = 50
LR = 0.00005

if __name__ == "__main__":
    # 입력 디렉토리에서 모든 jpg, png 파일을 처리
    torch.cuda.empty_cache()

    #학습된 모델 존재시 True 아닐시 False
    weights = os.path.exists(save_dir + '/model.pth')

    if weights:
        model = DenseUNet(num_classes=30, weights=False)
        model.load_state_dict(torch.load(save_dir + '/model.pth', weights_only=True))
    else:
        model = DenseUNet(num_classes=30, weights=True)
    model.to(device)

    dataset = CustomDataset(image_dir=data_dir, label_dir=label_dir)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [700, len(dataset) - 700])

    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    train_model(model, dataloader, epochs=50, learning_rate=LR)

    predict(model, test_dataset, device)
