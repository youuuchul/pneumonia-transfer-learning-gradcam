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

---

# 회고
- 모델에 대한 이해도는 조금 올랐는데, 아직 모델과 평가함수의 유기적인 코드 구성은 헷갈린다...
- 로거 써보긴 했는데 코드 복잡도가 올라가고, 통제가 좀 떨어진 듯 (ERROR 기능 안쓰고 로거 쓴 부분들도 있는 것 같기도..)
- 코랩으로 돌렸었는데 런타임 제한이 맥북 로컬의 느린속도보다 더 치명적이었다. 체크포인트 같은 걸로 코드 수정은 했지만 그래도 여전히..(체크포인트 저장 불러오기가 계정별로 다른 드라이브 경로에 저장되기도 했고..) -> 결국 로컬로 마무리했다.
- 이번에 코랩 에이전트 모드도 활용해봤는데 나름 편했고, ipynb 내에서만 실행하며 수정하는 점은 vscode 에이전트 모드와 차별점인듯
- Grad-cam 시도해봤는데, 코드도 해석도 난이도가 좀 있었음
