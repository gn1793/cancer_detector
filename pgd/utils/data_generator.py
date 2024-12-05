import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

class BloodCellDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.classes = ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']
        
        # 이미지 경로와 라벨 수집
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(data_path, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 이미지 로드 및 전처리
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def prepare_data(base_dir):
    # 데이터 디렉토리 설정
    data_dir = os.path.join(base_dir, 'datasets')
    source_dir = os.path.join(data_dir, 'Blood cell Cancer [ALL]')
    
    # 저장 경로 설정
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # 디렉토리 생성
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 클래스별 데이터 수집 및 분할
    for class_name in ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']:
        source_class_dir = os.path.join(source_dir, f'[Malignant] {class_name.split("_")[-1]}')
        if not os.path.exists(source_class_dir):
            print(f"Warning: {source_class_dir} not found")
            continue
            
        # 이미지 파일 목록 수집
        images = [f for f in os.listdir(source_class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 데이터 분할 (train:val:test = 70:10:20)
        train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
        train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.125, random_state=42)
        
        # 각 분할된 데이터셋에 대해 이미지 복사
        for split_name, split_imgs, split_dir in [
            ('train', train_imgs, train_dir),
            ('val', val_imgs, val_dir),
            ('test', test_imgs, test_dir)
        ]:
            # 클래스별 디렉토리 생성
            split_class_dir = os.path.join(split_dir, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            
            # 이미지 복사
            for img in tqdm(split_imgs, desc=f'Copying {split_name} {class_name}'):
                src = os.path.join(source_class_dir, img)
                dst = os.path.join(split_class_dir, img)
                shutil.copy2(src, dst)

if __name__ == "__main__":
    # 프로젝트 루트 디렉토리 설정
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # 데이터 준비
    prepare_data(project_root)
    
    print("데이터셋 준비가 완료되었습니다.")
