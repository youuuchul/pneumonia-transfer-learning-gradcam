# 🩺 Pneumonia Diagnosis with Transfer Learning & Grad-CAM  
### 전이학습 기반 폐렴(X-ray) 분류 및 시각적 해석 프로젝트  

---

## 📘 Overview | 프로젝트 개요

이 프로젝트는 **흉부 X-ray 이미지 데이터를 활용하여 폐렴 여부를 진단하는 분류 모델을 구축**하고,  
**Grad-CAM** 기반의 시각적 해석 기법을 통해 모델의 판단 근거를 설명하는 데 목적.

Transfer Learning(전이학습)을 활용하여 학습 효율을 높이고, 의료 이미지에서 신뢰성 있는 예측을 수행할 수 있도록 모델을 구성.

This project builds a **pneumonia classification model using chest X-ray images** and provides  
**visual explainability via Grad-CAM** to interpret how the model makes predictions.  
Using transfer learning significantly improves performance on limited medical datasets.

---

## 🗂 Dataset | 데이터셋

### 📌 Dataset: Chest X-Ray Pneumonia (Kaggle)  
🔗 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### 구성  
- **Train**  
  - NORMAL: 1,341 images  
  - PNEUMONIA: 3,875 images  
- **Test**  
  - NORMAL: 234 images  
  - PNEUMONIA: 390 images  
- 이미지 형태: `RGB / 3채널`, 다양한 해상도  
- 데이터 불균형 존재 → Weighted Loss 적용

### Summary  
- Real clinical X-ray images  
- Binary classification: NORMAL vs PNEUMONIA  
- Imbalanced dataset → class weighting & data augmentation applied  


---

## 🔍 Grad-CAM Visual Explanation | Grad-CAM 시각적 해석

**Grad-CAM**을 통해 모델이 어떤 영역을 근거로 폐렴을 판단했는지 확인했습니다.

- 폐렴이 있는 경우 → 염증이 있는 폐부 중심으로 activation 집중  
- 정상 이미지 → 비교적 넓고 분산된 activation  
- 과적합 여부 점검 가능  
