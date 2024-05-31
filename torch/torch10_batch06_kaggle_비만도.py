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
path = 'c:/_data/kaggle/Obesity_Risk/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

from sklearn.preprocessing import LabelEncoder

lae_G = LabelEncoder()
train_csv['Gender'] = lae_G.fit_transform(train_csv['Gender'])
test_csv['Gender'] = lae_G.transform(test_csv['Gender'])

lae_fhwo = LabelEncoder()
train_csv['family_history_with_overweight'] = lae_fhwo.fit_transform(train_csv['family_history_with_overweight'])
test_csv['family_history_with_overweight'] = lae_fhwo.transform(test_csv['family_history_with_overweight'])

lae_FAVC = LabelEncoder()
train_csv['FAVC'] = lae_FAVC.fit_transform(train_csv['FAVC'])
test_csv['FAVC'] = lae_FAVC.transform(test_csv['FAVC'])

lae_CAEC = LabelEncoder()
train_csv['CAEC'] = lae_CAEC.fit_transform(train_csv['CAEC'])
test_csv['CAEC'] = lae_CAEC.transform(test_csv['CAEC'])

lae_SMOKE = LabelEncoder()
train_csv['SMOKE'] = lae_SMOKE.fit_transform(train_csv['SMOKE'])
test_csv['SMOKE'] = lae_SMOKE.transform(test_csv['SMOKE'])

lae_SCC = LabelEncoder()
train_csv['SCC'] = lae_SCC.fit_transform(train_csv['SCC'])
test_csv['SCC'] = lae_SCC.fit_transform(test_csv['SCC'])

lae_CALC = LabelEncoder()
test_csv['CALC'] = lae_CALC.fit_transform(test_csv['CALC'])
train_csv['CALC'] = lae_CALC.transform(train_csv['CALC'])

lae_MTRANS = LabelEncoder()
train_csv['MTRANS'] = lae_MTRANS.fit_transform(train_csv['MTRANS'])
test_csv['MTRANS'] = lae_MTRANS.transform(test_csv['MTRANS'])

lae_NObeyesdad = LabelEncoder()
train_csv['NObeyesdad'] = lae_NObeyesdad.fit_transform(train_csv['NObeyesdad'])

x = train_csv.drop(['NObeyesdad'], axis = 1)
y = train_csv['NObeyesdad']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

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

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

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
    
model = Model(16, 7).to(DEVICE)  # (input_dim, output_dim)

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

epochs = 100
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

# 최종 loss :  0.48283402018446836
# [2 1 1 ... 0 1 1]
# Accuracy: 0.8281

