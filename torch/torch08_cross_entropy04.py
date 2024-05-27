# 04. dacon wine
# 05. 대출
# 06. kaggle 비만도
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# GPU 사용 여부 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)

# 1. 데이터 로드 및 전처리
path = "c:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)    # [5497 rows x 13 columns]
print(train_csv.shape)  # (5497, 13)
print(train_csv.head) 

test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
print(test_csv)     
print(test_csv.shape)  # (1000, 12)
print(test_csv.info()) 

submission_csv = pd.read_csv(path + 'sample_submission.csv')
print(submission_csv)
print(submission_csv.shape)  # (1000, 2)

train_csv['type'] = train_csv['type'].map({"white":1, "red":0}).astype(int)
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0}).astype(int)

# x와 y를 분리
x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']
print(x.shape, y.shape)     # (5497, 12) (5497,)
print(np.unique(y, return_counts=True)) # (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))
print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train.values - 3).to(DEVICE)  # 라벨 범위 조정
y_test = torch.LongTensor(y_test.values - 3).to(DEVICE)    # 라벨 범위 조정

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2. 모델 구성
model = nn.Sequential(
    nn.Linear(12, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32, 7) 
).to(DEVICE)

# 3. 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습 함수
def train(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    return loss.item()

# 학습
epochs = 4000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    if epoch % 100 == 0 or epoch == 1:
        print(f'epoch: {epoch}, loss: {loss}')

print('==========================================================')

# 모델 평가 함수
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    with torch.no_grad():
        y_predict = model(x_test)
        loss2 = criterion(y_predict, y_test)
    return loss2.item(), y_predict

# 평가 및 성능 측정
loss2, y_predict = evaluate(model, criterion, x_test, y_test)
print('최종 loss:', loss2)

# 성능 평가
y_predict = model(x_test).cpu().detach().numpy()

# 예측값을 클래스 레이블로 변환
y_predict = np.argmax(y_predict, axis=1)

accuracy = accuracy_score(y_test.cpu().numpy(), y_predict)
precision = precision_score(y_test.cpu().numpy(), y_predict, average='macro')
recall = recall_score(y_test.cpu().numpy(), y_predict, average='macro')
f1 = f1_score(y_test.cpu().numpy(), y_predict, average='macro')

print('Accuracy: {:.4f}'.format(accuracy))
print('Precision: {:.4f}'.format(precision))
print('Recall: {:.4f}'.format(recall))
print('F1 Score: {:.4f}'.format(f1))

# Accuracy: 0.5900
# Precision: 0.3858
# Recall: 0.3003
# F1 Score: 0.3073
