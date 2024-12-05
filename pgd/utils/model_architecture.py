import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import json

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        return self.gamma * out + x

class BloodCellNet(nn.Module):
    def __init__(self):
        super(BloodCellNet, self).__init__()
        self.features = nn.Sequential(
            # 첫 번째 블록
            nn.Conv2d(3, 128, kernel_size=8, stride=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            SelfAttention(128),
            
            # 두 번째 블록
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3),
            SelfAttention(256),
            
            # 세 번째 블록부터 일곱 번째 블록까지
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            SelfAttention(256),
            
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            SelfAttention(512),
            
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            SelfAttention(512)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def save_model_architecture():
    # 모델 구조를 딕셔너리로 정의
    model_architecture = {
        'name': 'BloodCellNet',
        'input_shape': [3, 224, 224],
        'features': {
            'conv_layers': [
                {'type': 'Conv2D', 'filters': 128, 'kernel_size': 8, 'stride': 3},
                {'type': 'ReLU'},
                {'type': 'BatchNorm2D', 'num_features': 128},
                {'type': 'SelfAttention', 'in_channels': 128},
                
                {'type': 'Conv2D', 'filters': 256, 'kernel_size': 5, 'stride': 1},
                {'type': 'ReLU'},
                {'type': 'BatchNorm2D', 'num_features': 256},
                {'type': 'MaxPool2D', 'size': 3},
                {'type': 'SelfAttention', 'in_channels': 256},
                # ... 나머지 레이어 정보
            ]
        },
        'classifier': {
            'layers': [
                {'type': 'Linear', 'in_features': 512 * 3 * 3, 'out_features': 2048},
                {'type': 'ReLU'},
                {'type': 'Dropout', 'p': 0.2},
                {'type': 'Linear', 'in_features': 2048, 'out_features': 1024},
                {'type': 'ReLU'},
                {'type': 'Dropout', 'p': 0.2},
                {'type': 'Linear', 'in_features': 1024, 'out_features': 512},
                {'type': 'ReLU'},
                {'type': 'Dropout', 'p': 0.2},
                {'type': 'Linear', 'in_features': 512, 'out_features': 4},
                {'type': 'Softmax'}
            ]
        }
    }

    # config 디렉토리에 저장하기 위한 경로 설정
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'architecture.h5')
    
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # .h5 파일로 저장
    with h5py.File(save_path, 'w') as f:
        # 모델 구조를 JSON 문자열로 변환하여 저장
        architecture_str = json.dumps(model_architecture)
        f.create_dataset('model_architecture', data=architecture_str.encode('utf-8'))
        
        # 모델의 기본 속성 저장
        f.attrs['model_name'] = 'BloodCellNet'
        f.attrs['input_shape'] = [3, 224, 224]
        f.attrs['output_classes'] = 4

    print(f"모델 아키텍처가 '{save_path}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    save_model_architecture()
