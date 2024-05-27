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
path= "c:\_data\dacon\dechul\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_csv = pd.read_csv(path + "sample_submission.csv")

# 라벨 엔코더
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder() # 대출기간, 대출목적, 근로기간, 주택소유상태 // 라벨 인코더 : 카테고리형 피처를 숫자형으로 변환
train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = le.fit_transform(train_csv['대출목적'])
train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])

test_csv['주택소유상태'] = le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = le.fit_transform(test_csv['대출목적'])
test_csv['대출기간'] = test_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
test_csv['근로기간'] = le.fit_transform(test_csv['근로기간'])

train_csv['대출등급'] = le.fit_transform(train_csv['대출등급']) # 마지막에 와야함

print(train_csv.describe)
print(test_csv.describe)

print(train_csv.shape)
print(test_csv.shape)
print(train_csv.dtypes)
print(test_csv.dtypes)
# x와 y를 분리
x = train_csv.drop(['대출등급','총계좌수'], axis=1)
y = train_csv['대출등급']
test_csv = test_csv.drop(['총계좌수'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train.values).to(DEVICE)  # 라벨 범위 조정
y_test = torch.LongTensor(y_test.values).to(DEVICE)    # 라벨 범위 조정

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

# 최종 loss: 0.4066654443740845
# Accuracy: 0.8612
# Precision: 0.8567
# Recall: 0.7327
# F1 Score: 0.7344
