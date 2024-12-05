import io
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from torchvision import transforms
import os

# Flask 애플리케이션 초기화 및 CORS 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(os.path.dirname(current_dir), 'templates')
static_dir = os.path.join(os.path.dirname(current_dir), 'static')

app = Flask(__name__, 
    template_folder=template_dir,
    static_url_path='/static',
    static_folder=static_dir
)
CORS(app)

# 분류할 혈액 세포 클래스 정의
CLASSES = ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']

# Self-Attention 메커니즘 구현
# 이미지의 중요한 특징을 자동으로 집중하여 학습하는 모듈
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # Query, Key, Value 변환을 위한 1x1 합성곱 레이어
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 학습 가능한 가중치
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        # Query, Key, Value 계산
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        
        # Attention 점수 계산 및 softmax 적용
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Attention 가중치를 Value에 적용
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # residual connection
        return self.gamma * out + x

# 혈액 세포 분류를 위한 CNN 모델 정의
class BloodCellNet(nn.Module):
    def __init__(self):
        super(BloodCellNet, self).__init__()
        
        # 특징 추출을 위한 CNN 레이어들
        self.features = nn.Sequential(
            # 첫 번째 블록: 입력 이미지 처리
            nn.Conv2d(3, 128, kernel_size=8, stride=3),  # RGB 3채널 입력
            nn.ReLU(),  # 활성화 함수
            nn.BatchNorm2d(128),  # 배치 정규화
            SelfAttention(128),  # 어텐션 메커니즘
            
            # 두 번째 블록: 특징 맵 확장
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3),  # 특징 맵 크기 축소
            SelfAttention(256),
            
            # 세 번째 블록: 특징 추출 심화
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            SelfAttention(256),
            
            # 네 번째 블록: 1x1 합성곱으로 채널 간 ��보 통합
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            # 다섯 번째 블록: 채널 수 증가
            nn.Conv2d(256, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            SelfAttention(512),
            
            # 여섯 번째 블록: 고수준 특징 추출
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            
            # 일곱 번째 블록: 최종 특징 추출
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            SelfAttention(512)
        )
        
        # 분류를 위한 완전연결층
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 2D 특징 맵을 1D로 평탄화
            nn.Linear(512 * 3 * 3, 2048),  # 첫 번째 완전연결층
            nn.ReLU(),
            nn.Dropout(0.2),  # 과적합 방지
            nn.Linear(2048, 1024),  # 두 째 완전연결층
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),  # 세 번째 완전연���층
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, len(CLASSES)),  # 출력층 (클래스 수만큼)
            nn.Softmax(dim=1)  # 확률 분포로 변환
        )

    def forward(self, x):
        x = self.features(x)  # 특징 추출
        x = self.classifier(x)  # 분류
        return x

# 상대 경로를 사용하여 모델 파일 위치 지정
current_dir = os.path.dirname(os.path.abspath(__file__))  # web/app 디렉토리
project_root = os.path.dirname(os.path.dirname(current_dir))  # pgd 디렉토리
model_path = os.path.join(project_root, 'checkpoints', 'best_model.pth')

model = BloodCellNet()

try:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
except FileNotFoundError:
    raise FileNotFoundError(f"모델 파일을 {model_path}에서 찾을 수 없습니다. 'best_model.pth'가 checkpoints 디렉토리에 있는지 확인하세요.")

model.eval()  # 평가 모드로 설정

# 이미지 전처리를 위한 변환 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 통일
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# 메인 ��이지 라우트
@app.route('/')
def index():
    return render_template('index.html')

# 이미지 분류 API 엔드포인트
@app.route('/api/classify', methods=['POST'])
def classify_image():
    # 파일 업로드 확인
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'}), 400
     
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없니다'}), 400
    
    if file:
        # 업로드된 파일을 이미지로 변환
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 이미지 전처리
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # 배치 차원 추가
        
        # 예측 행
        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_class = CLASSES[torch.argmax(outputs[0])]
        
        # 예측 결과 반환
        return jsonify({'prediction': predicted_class})

# 메인 실행 부분
if __name__ == '__main__':
    app.run(debug=True)  # 개발 모드로 서버 실행
