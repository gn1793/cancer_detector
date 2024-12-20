# 설치 방법

## 1. 프로젝트 클론
   
   git clone https://github.com/gn1793/cancer_detector.git
   
   cd cancer_detector

## 2. 가상환경 생성 및 활성화
가상환경 생성
python -m venv venv

가상환경 활성화
Windows:
venv\Scripts\activate
macOS/Linux:
source venv/bin/activate


## 3. 의존성 패키지 설치
pip install -r requirements.txt

## 4. 데이터셋 다운로드 및 설치
   - 원본 데이터셋: https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class
   - 전처리된 데이터셋: https://drive.google.com/drive/folders/16epgHrJZCAthtwJHxeBliov2hU748LtD?usp=drive_link
   - 체크포인트 : https://drive.google.com/drive/folders/1K_hjwTKzeZ2MgQD3eQVjTfz09BF173zy?usp=drive_link
   
   전처리된 데이터셋을 다운로드 받은 후, 프로젝트의 datasets 폴더 안에 train, val, test 폴더를 각각 위치시켜주세요.
   
   체크포인트 파일(best_model.pth)을 다운로드 받은 후, pgd의 checkpoints 폴더 안에 위치시켜주세요.
   
   최종적으로 프로젝트 구조와 동일하게 위치시켜주세요.
## 데이터셋 구조
```
0 : Benign
1 : Malignant_Pre-B
2 : Malignant_Pro-B
3 : Malignant_early Pre-B
```

## 프로젝트 구조
```
cancer_detector/
├── models/
│   └── blood_cell_net.py    
├── web/
│   ├── model.py           
│   ├── page.py           
│   └── templates/
│       └── index.html
├── datasets/
│   ├── train/
│   ├── val/
│   └── test/
├── checkpoints/
│   └── best_model.pth
└── requirements.txt
```
# 실행 방법

## 1. 모델 학습:
python -m models.blood_cell_net

## 2. 웹 서버 실행:
python -m web.page

## 3. 웹 서버 이용:
``` datasets/test의 데이터를 이용해 웹 실행 ```
