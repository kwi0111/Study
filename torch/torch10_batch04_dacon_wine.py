import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# GPU 사용 가능 여부 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ' , torch.__version__ , '사용 DEVICE : ', DEVICE)

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
y = train_csv['quality'] - 3
print(x.shape, y.shape)     # (5497, 12) (5497,)
print(np.unique(y, return_counts=True)) # (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))
print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train.values).to(DEVICE)
y_test = torch.LongTensor(y_test.values).to(DEVICE)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 토치 데이터 만들기
from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)
print(train_set)

print(type(train_set))
print(len(train_set))

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# 모델구성
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear5(x)
        return x
    
model = Model(12, 7).to(DEVICE)  # (input_dim, output_dim)

# 3. 컴파일
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, loader):
    total_loss = 0
    model.train()
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

epochs = 300
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    if epoch % 5 == 0:
        print('epoch : {} , loss:{}'.format(epoch, loss))

print('==========================================================')

def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_predict = model(x_batch)
            loss = criterion(y_predict, y_batch)
            total_loss += loss.item()
            y_pred_list.append(y_predict)
            y_true_list.append(y_batch)
    y_pred_list = torch.cat(y_pred_list).cpu().detach().numpy()
    y_true_list = torch.cat(y_true_list).cpu().detach().numpy()
    return total_loss / len(loader), y_pred_list, y_true_list

loss2, y_pred, y_true = evaluate(model, criterion, test_loader)
print('최종 loss : ', loss2)

y_pred = np.argmax(y_pred, axis=1)
print(y_pred)

score = accuracy_score(y_true, y_pred)
print('Accuracy: {:.4f}'.format(score))  

# 최종 loss :  1.0635506705595896
# [4 2 3 ... 3 2 4]
# Accuracy: 0.5964

