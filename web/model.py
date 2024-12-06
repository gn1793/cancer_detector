import os
import io
from PIL import Image
import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# 절대 경로로 import 수정
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from models.blood_cell_net import BloodCellNet

# FastAPI 앱 초기화
app = FastAPI(
    title="Blood Cell Cancer Classifier API",
    description="혈액 세포 이미지를 분석하여 암 세포 유형을 분류하는 API",
    version="1.0.0"
)

# 상수 정의
CLASSES = ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']

# 모델 로드
try:
    # 체크포인트 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(project_root, 'checkpoints', 'best_model.pth')
    
    print(f"현재 디렉토리: {current_dir}")
    print(f"프로젝트 루트: {project_root}")
    print(f"체크포인트 경로: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"체크포인트 파일이 존재하지 않습니다: {checkpoint_path}")
    
    # 모델 초기화 및 가중치 로드
    model = BloodCellNet(num_classes=len(CLASSES))
    print("모델 초기화 완료")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    print("체크포인트 로드 완료")
    print(f"체크포인트 키: {checkpoint.keys()}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("모델 가중치 로드 완료")
    
    model.eval()
    print(f"모델 로드 성공: {checkpoint_path}")

except Exception as e:
    print(f"모델 로드 실패 - 상세 에러: {str(e)}")
    print(f"에러 타입: {type(e)}")
    import traceback
    print(f"스택 트레이스: {traceback.format_exc()}")
    model = None

# 이미지 전처리 파이프라인 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 모델 입력 크기에 맞게 조정
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet 평균
        std=[0.229, 0.224, 0.225]    # ImageNet 표준편차
    )
])

@app.post("/api/classify", response_model=dict)
async def classify_image(file: UploadFile = File(...)):
    try:
        # 파일 검증
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
        
        # 이미지 로드 및 전처리
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  #  차원 추가
        
        # 예측 수행
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = float(probabilities[predicted_class_idx])
        
        # 결과 반환
        return {
            "prediction": CLASSES[predicted_class_idx],
            "confidence": confidence,
            "probabilities": {
                class_name: float(prob)
                for class_name, prob in zip(CLASSES, probabilities)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")

# 서버 상태 확인용 엔드포인트
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
