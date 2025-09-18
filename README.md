# Segmentation_Model

**컴퓨터공학 / 딥러닝 프로젝트**

UNet 기반 세그멘테이션 모델을 구현하여 이미지 내 객체를 픽셀 단위로 분할(Semantic Segmentation)하는 프로젝트입니다.  
특히 **Dense-UNet**과 **RD-UNet** 두 가지 변형 구조를 적용해 성능을 비교했습니다.

---

## 🎯 프로젝트 목표

- 시맨틱 세그멘테이션 모델 학습 및 구현
- DenseNet / Residual DenseNet 기반 Encoder 실험
- IoU, Loss 기반 성능 비교
- 실제 예측 결과를 시각화

---

## ⚙️ 개발 환경

- **OS**: Windows 11  
- **CPU**: AMD Ryzen 5 5600 (6-Core)  
- **GPU**: NVIDIA GeForce RTX 4060  
- **RAM**: 16GB  

- **Python**: 3.9.19  
- **CUDA**: 12.6  
- **cuDNN**: 9.5.1  

- **Dependencies**:  
  ```
  torch==2.5.1
  torchvision==0.20.1
  timm==1.0.12
  pandas==2.2.3
  numpy==2.0.1
  pillow==11.0.0
  ```

---

## 🧩 모델 구조

### 1. Dense-UNet
- Encoder에 **DenseNet-121 (pretrained)** 적용  
- DenseNet은 feature 재사용 구조 → **적은 데이터셋에도 효율적**  
- 원 논문은 Upsampling에도 DenseBlock을 사용하지만, 리소스 한계로 Encoder에만 적용  
- **Loss**: Dice Loss (Cityscapes의 클래스 불균형 대응)

**결과**  
- 최소 Loss: `0.682`  
- 최대 IoU: `0.385`  
- 파라미터 수: 13,937,152  
- GFLOPs: 489.89  
- 메모리 사용량: 18.2GB

---

### 2. RD-UNet
- Encoder를 **RDNet-T**로 교체  
- Dense Connection + Residual Connection 결합  
- Stage 내부는 Dense, Stage 간에는 Residual 연결  
- **EffectiveSEModule** 추가 → 채널 간 중요도 조정  
- 참고: [ECCV 2024 DenseNets Reloaded (naver-ai/rdnet)]

**결과**  
- 최소 Loss: `0.614`  
- 최대 IoU: `0.430`  
- 파라미터 수: 131,293,992  
- GFLOPs: 153.35  
- 메모리 사용량: 2.3GB

---

## 🚀 실행 방법

1. **데이터셋 준비**
   ```
   dataset/
   ├── images/
   │    ├── 1.jpg
   │    └── 2.jpg
   └── masks/
        ├── 1.png
        └── 2.png
   ```

2. **Dense-UNet 학습**
   ```bash
   python Dense_UNet.py --epochs 50 --batch_size 8 --lr 1e-4
   ```

3. **RD-UNet 학습**
   ```bash
   python RD_Unet.py --epochs 50 --batch_size 8 --lr 1e-4
   ```

4. **결과 확인**
   - 학습된 모델 불러오기
   - 테스트 이미지 분할 후 마스크 저장

---

## 📊 성능 비교

| 모델       | 최소 Loss | 최대 IoU | 파라미터 수     | GFLOPs | 메모리 사용량 |
|------------|-----------|----------|-----------------|--------|---------------|
| Dense-UNet | 0.682     | 0.385    | 13.9M           | 490    | 18.2GB        |
| RD-UNet    | 0.614     | 0.430    | 131.3M          | 153    | 2.3GB         |

---

## 🖼️ 결과 예시

### 예시 1
| 입력 이미지 | 예측 마스크 (Gray) | 예측 마스크 (Color) |
|-------------|-------------------|---------------------|
| ![input1](docs/input_image.jpg) | ![mask_gray1](docs/predicted_image8.jpg) | ![mask_color1](docs/predicted_color_image8.jpg) |

### 예시 2
| 입력 이미지 | 예측 마스크 (Gray) | 예측 마스크 (Color) |
|-------------|-------------------|---------------------|
| ![input2](docs/input2.jpg) | ![mask_gray2](docs/predicted_1366_2000.jpg) | ![mask_color2](docs/predicted_color_1366_2000.jpg) |

---

## 📚 배운 점

- DenseNet과 Residual DenseNet의 Encoder 구조 차이가 성능에 미치는 영향 이해
- 데이터 불균형 대응을 위해 Dice Loss를 적용한 효과 확인
- 모델 파라미터 수와 FLOPs, 메모리 사용량이 실제 학습 환경에 큰 영향을 미침
- Segmentation의 응용 가능성(자율주행, 의료 영상, 위성 이미지 분석 등)을 체감

---

## 📄 라이선스

학습 및 연구 목적으로 작성된 프로젝트입니다.  
자유롭게 참고 가능하며, 사용 시 출처를 명시해 주세요.
