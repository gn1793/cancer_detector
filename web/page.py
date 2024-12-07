import os
import sys
from contextlib import redirect_stdout
import io

# blood_cell_net.py의 출력을 숨기기 위해 임시로 stdout 리다이렉트
with redirect_stdout(io.StringIO()):
    from models.blood_cell_net import BloodCellNet

import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from torchvision import transforms
from PIL import Image

# Flask 애플리케이션 초기화 및 CORS 설정
current_dir = os.path.dirname(os.path.abspath(__file__))  # web 폴더
project_root = os.path.dirname(current_dir)  # cancer_detector 폴더
template_dir = os.path.join(current_dir, 'templates')
static_dir = os.path.join(current_dir, 'static')

app = Flask(__name__, 
    template_folder=template_dir,
    static_url_path='/static',
    static_folder=static_dir
)
CORS(app)

# 분류할 혈액 세포 클래스 정의
CLASSES = ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']

# 모델 로드
model_path = os.path.join(project_root, 'checkpoints', 'best_model.pth')
model = BloodCellNet()

try:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
except FileNotFoundError:
    raise FileNotFoundError(f"모델 파일을 {model_path}에서 찾을 수 없습니다.")

model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'}), 400
     
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다'}), 400
    
    if file:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_class = CLASSES[torch.argmax(outputs[0])]
        
        return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
