import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# GPU 사용 가능 여부 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)

#1. 데이터 (분석, 정제, 전처리) // 판다스 
path = "C:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)      # 날짜 데이터 인덱스로
submission_csv = pd.read_csv(path + "samplesubmission.csv")

# 결측치 처리
print(train_csv.isna().sum())           # 데이터 프레임 결측치  없다.

# x와 y를 분리
x = train_csv.drop(['casual','registered','count'], axis=1)       # 행삭제 : axis = 0 // 열삭제 : axis = 1 // train_csv에 있는 'count'열 삭제 
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train.values).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test.values).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 클래스로 바꿀거임
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):  
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
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
    
model = Model(8, 1).to(DEVICE)  # (input_dim, output_dim)

# 3. 컴파일
criterion = nn.MSELoss()            
optimizer = optim.Adam(model.parameters(), lr=0.01)  

def train(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()     
    hypothesis = model(x)  # 예상치 값 (순전파)
    loss = criterion(hypothesis, y)  # 예상값과 실제값 loss
    loss.backward()  # 기울기(gradient)값(loss를 weight로 미분한 값) 계산 -> 역전파 시작
    optimizer.step()  # 가중치 수정(w 갱신) -> 역전파 끝
    return loss.item()  # item을 쓰는 이유는 numpy 데이터로 뽑기위해서 똑같이 tensor 데이터는 맞음
    
epochs = 2000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    if epoch % 100 == 0:  # 에포크마다 출력
        print('epoch : {} , loss:{}'.format(epoch, loss))

print('==========================================================')

# 4. 평가, 예측
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    with torch.no_grad():
        y_predict = model(x_test)
        loss = criterion(y_predict, y_test)
    return loss.item(), y_predict

loss2, y_predict = evaluate(model, criterion, x_test, y_test)
print('최종 loss : ', loss2)

y_predict = y_predict.cpu().detach().numpy()
y_test = y_test.cpu().detach().numpy()

mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('MSE: {:.4f}'.format(mse))
print('R2 Score: {:.4f}'.format(r2))


# 최종 loss :  21347.521484375
# MSE: 21347.5195
# R2 Score: 0.3532