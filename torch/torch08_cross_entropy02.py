import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# GPU 사용 여부 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)

# 1. 데이터 로드 및 전처리
dataset = load_digits()
x = dataset.data
y = dataset.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2. 모델 구성
model = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32, 10)  # 10개의 클래스에 맞게 출력 크기를 설정합니다.
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
epochs = 2000
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

# 최종 loss: 0.176703542470932
# Accuracy: 0.9889
# Precision: 0.9893
# Recall: 0.9887
# F1 Score: 0.9887
