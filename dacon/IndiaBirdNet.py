import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torchvision.models as models


import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore') 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import torch
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("CUDA Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))


path = 'C:\\_data\\dacon\\Bird\\'

CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 100,
    'LEARNING_RATE': 0.0001,
    'BATCH_SIZE': 42,
    'SEED': 42,
    'PATIENCE': 5,  # 얼리 스톱핑을 위한 인내심 설정
}

def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_score = 0
    best_model = None
    patience_counter = 0  # 얼리 스톱핑 카운터
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()  # 기존 코드에서 누락된 부분을 추가합니다.
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        
        print(f'Epoch [{epoch}], Train Loss: {_train_loss:.5f}, Val Loss: {_val_loss:.5f}, Val F1 Score: {_val_score:.5f}')
       
        if scheduler is not None:
            scheduler.step(_val_score)
            
        # 성능이 개선되었는지 확인
        if best_score < _val_score:
            best_score = _val_score
            best_model = model
            patience_counter = 0  # 성능이 개선되었으므로 카운터를 리셋합니다.
        else:
            patience_counter += 1  # 성능이 개선되지 않았으므로 카운터를 증가시킵니다.
        
        # 얼리 스톱핑 조건 체크
        if patience_counter > CFG['PATIENCE']:
            print("Early stopping triggered.")
            break
    
    return best_model
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
seed_everything(CFG['SEED']) # Seed 고정

df = pd.read_csv(path+'train.csv')
train, val, _, _ = train_test_split(df, df['label'], test_size=0.25, stratify=df['label'], random_state=CFG['SEED'])

le = preprocessing.LabelEncoder()
train['label'] = le.fit_transform(train['label'])
val['label'] = le.transform(val['label'])

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img_path = img_path.replace('./', '')  # './' 제거
        img_path = os.path.join('C:/_data/dacon/Bird/', img_path)  # 절대 경로로 변경
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"Cannot load image at {img_path}")

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']  # 이 부분에서 텐서로 변환됩니다.


        # label_list가 None이 아닐 때만 레이블 처리
        if self.label_list is not None:
            label = self.label_list[index]
        else:
            # 레이블이 없는 경우 (예: 테스트 데이터셋) 임의의 값을 할당하거나, 레이블 관련 처리를 생략
            label = -1  # 또는 label = None 등, 처리 방식에 따라 변경 가능

        return image, torch.tensor(label, dtype=torch.long)


    def __len__(self):
        return len(self.img_path_list)
    

train_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.HorizontalFlip(p=0.2),  # 이미지를 수평으로 뒤집기
    A.VerticalFlip(p=0.2),  # 추가: 이미지를 수직으로 뒤집기
    A.RandomRotate90(p=0.3),  # 추가: 이미지를 90도 단위로 무작위 회전
    A.Rotate(limit=15),  # 이미지 회전
    A.GaussianBlur(p=0.05),  # 가우시안 블러 적용
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),  # 추가: 색상 조정
    A.OneOf([  # 추가: 이 중 하나의 변환을 적용
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
    ], p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
])

test_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

train_dataset = CustomDataset(train['img_path'].values, train['label'].values, train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

val_dataset = CustomDataset(val['img_path'].values, val['label'].values, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

# class BaseModel(nn.Module):
#     def __init__(self, num_classes=len(le.classes_)):
#         super(BaseModel, self).__init__()
#         self.backbone = models.efficientnet_b1(pretrained=True)
#         self.classifier = nn.Linear(1000, num_classes)
        
#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.classifier(x)
#         return x
    
def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_score = 0
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for imgs, labels in train_loader:
            imgs = imgs.to(device)  # 이미지 데이터를 GPU로 이동
            labels = labels.to(device)  # 레이블 데이터를 GPU로 이동

            
            output = model(imgs)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 Score : [{_val_score:.5f}]')
       
        if scheduler is not None:
            scheduler.step(_val_score)
            
        if best_score < _val_score:
            best_score = _val_score
            best_model = model
    
    return best_model
def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)  # 이미지 데이터를 GPU로 이동
            labels = labels.to(device)  # 레이블 데이터를 GPU로 이동
            
            pred = model(imgs)
            
            loss = criterion(pred, labels)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
            
            val_loss.append(loss.item())
        
        _val_loss = np.mean(val_loss)
        _val_score = f1_score(true_labels, preds, average='macro')
    
    return _val_loss, _val_score

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        # EfficientNet B0을 backbone으로 사용
        self.backbone = models.efficientnet_b1(pretrained=True)
        # EfficientNet B0의 분류기를 제거 (특징 추출기만 사용)
        self.features = self.backbone.features
        
        # 추가할 배치 정규화 레이어와 분류기
        self.batchnorm = nn.BatchNorm1d(1280)  # EfficientNet B0의 특징 벡터 크기에 맞춤
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),  # 1280은 EfficientNet B0의 특징 벡터 크기
            nn.ReLU(),
            nn.BatchNorm1d(512),  # 추가된 배치 정규화 레이어
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)  # Global Average Pooling
        x = self.batchnorm(x)
        x = self.classifier(x)
        return x

# model = BaseModel()
# model = BaseModel(num_classes=10).to(device)
model = BaseModel(num_classes=len(le.classes_)).to(device)
model.eval()


optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold_mode='abs', min_lr=1e-8, verbose=True)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'], eta_min=0)

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

test = pd.read_csv(path + 'test.csv')
test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(iter(test_loader)):
            imgs = batch[0].to(device)  # 첫 번째 요소가 이미지 데이터라고 가정
            pred = model(imgs)
            preds += pred.argmax(1).detach().cpu().numpy().tolist()

    preds = le.inverse_transform(preds)
    return preds



preds = inference(infer_model, test_loader, device)

submit = pd.read_csv(path + 'sample_submission.csv')
submit['label'] = preds
submit.to_csv(path + 'submit.csv', index=False)    
