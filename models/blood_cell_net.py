import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action="ignore")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import sklearn.metrics as metrics

import os
import pandas as pd
from tqdm import tqdm
import time

import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)  
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1) 
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)       
        self.gamma = nn.Parameter(torch.zeros(1))  
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Q, K, V 변환 및 행렬 곱을 위한 reshape
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  
        key = self.key(x).view(batch_size, -1, height * width)                        
        value = self.value(x).view(batch_size, -1, height * width)                   
        
        # Attention 점수 계산
        attention = torch.bmm(query, key)  
        attention = F.softmax(attention, dim=-1)  # 각 위치별 attention 점수 정규화
        
        # Attention & Value 결합
        out = torch.bmm(value, attention.permute(0, 2, 1))  
        out = out.view(batch_size, channels, height, width)  
        
        # Residual connection
        return self.gamma * out + x

# 현재 파일의 절대 경로를 기준으로 상대 경로 설정
current_file_path = os.path.dirname(os.path.abspath(__file__))  # models 폴더
project_root = os.path.dirname(current_file_path)  # cancer_detector 폴더
BASE_PATH = os.path.join(project_root, 'datasets')
TRAIN_PATH = os.path.join(BASE_PATH, 'train')
VAL_PATH = os.path.join(BASE_PATH, 'val')
TEST_PATH = os.path.join(BASE_PATH, 'test')

# 데이터셋 클래스 수정
class BloodCellDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.classes = ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']
        
        # 이미지 경로와 라벨 수집
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(data_path, str(class_idx))
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(class_idx)
        
        # 디버깅: 로드된 이미지 경로 수 출력
        print(f"Loaded {len(self.image_paths)} images from {data_path}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 데이터셋 생성
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = BloodCellDataset(TRAIN_PATH, transform=transform)
val_dataset = BloodCellDataset(VAL_PATH, transform=transform)
test_dataset = BloodCellDataset(TEST_PATH, transform=transform)

# 데이터로더 생성
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 데이터셋 크기 확인
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# 이미지 시각화 함수 수정
def show_Blood_images(loader):
    images, labels = next(iter(loader))
    plt.figure(figsize=(20,20))
    length = len(labels)
    if length < 25:
        r = length
    else:
        r = 25
    for i in range(r):
        plt.subplot(5,5,i+1)
        image = images[i].permute(1, 2, 0)  # CHW -> HWC
        image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        image = image.clip(0, 1)
        plt.imshow(image)
        class_name = train_dataset.classes[labels[i]]
        plt.title(class_name, color="green", fontsize=16)
        plt.axis('off')
    plt.show()

# 이미지 시각화
show_Blood_images(train_loader)


classes = ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']
print(classes)

# PyTorch 커스텀 데이터셋 클래스
class BloodCellDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.classes = ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepaths']
        label = self.classes.index(self.dataframe.iloc[idx]['labels'])
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 모델 정의
class BloodCellNet(nn.Module):
    def __init__(self, num_classes=4):
        super(BloodCellNet, self).__init__()
        
        # 특징 추출을 위한 CNN 레이어들
        self.features = nn.Sequential(
            # Block 1: 입력 처리 (224x224x3 -> 73x73x128)
            nn.Conv2d(3, 128, kernel_size=8, stride=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            SelfAttention(128),
            
            # Block 2: 특징 맵 확장 (73x73x128 -> 24x24x256)
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3),
            SelfAttention(256),
            
            # Block 3: 특징 추출 심화 (24x24x256 -> 24x24x256)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            SelfAttention(256),
            
            # Block 4: 채널 간 정보 통합 (24x24x256 -> 24x24x256)
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            # Block 5: 특징 맵 압축 (24x24x256 -> 12x12x512)
            nn.Conv2d(256, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            SelfAttention(512),
            
            # Block 6: 특징 맵 압축 (12x12x512 -> 6x6x512)
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            
            # Block 7: 최종 특징 추출 (6x6x512 -> 3x3x512)
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            SelfAttention(512)
        )
        
        # 분류를 위한 완전연결층
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),  # 과적합 방지
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)  # 클래스별 확률 출력
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 이미지 [batch_size, 3, 224, 224]
            
        Returns:
            torch.Tensor: 클래스별 확률 [batch_size, num_classes]
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

# 모델 학습 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BloodCellNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 학습 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30):
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        start_time = time.time()
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # 학습 진행률 표시
        train_pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 현재 배치의 손실과 정확도 표시
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*predicted.eq(labels).sum().item()/labels.size(0):.2f}%'
            })
        
        # 검증
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validation', leave=False)
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*predicted.eq(labels).sum().item()/labels.size(0):.2f}%'
                })
        
        # 결과 저장
        epoch_time = time.time() - start_time
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - {epoch_time:.1f}s - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%')
        
        # 검증 단계에서 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'train_acc': train_acc
            }
            torch.save(checkpoint, 'best_model.pth')
            print(f'Checkpoint saved! Val Loss improved from {best_val_loss:.4f} to {val_loss:.4f}')
    
    return history

# 모델 로드 함수
def load_model(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    print(f'Model loaded from epoch {epoch} with validation loss: {val_loss:.4f}')
    return model, optimizer, epoch

if __name__ == '__main__':
    # 학습 디렉토리 생성
    save_dir = 'plots'
    os.makedirs(save_dir, exist_ok=True)
    
    # 모델 학습
    history = train_model(model, train_loader, val_loader, criterion, optimizer)
    
    # 1. 학습 히스토리 그래프
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='train')
    plt.plot(history['val_acc'], label='validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history.png')
    plt.show()
    
    # 테스트 데이터에 대한 예측
    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_score.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)
    
    # 2. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.show()
    
    # 3. ROC Curve
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_curves.png')
    plt.show()
    
    # Classification Report 출력
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

