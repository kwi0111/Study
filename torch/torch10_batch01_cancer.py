import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# GPU 를 되는지 안되는지 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ' , torch.__version__ , '사용 DEVICE : ', DEVICE)


#1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 토치 데이터 만들기
#1. x와 y를 합친다!!
from torch.utils.data import TensorDataset
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)
print(train_set)    # <torch.utils.data.dataset.TensorDataset object at 0x0000020F20D194F0> // itrator 형식

#1. 배치 넣어준다.
print(type(train_set))
# print((train_set.shape)) # 에러
print(len(train_set))   # 398

#2. 배치 넣어준다. 끝
from torch.utils.data import DataLoader
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)    # test는 셔플할 이유는 없다. 통상 shuffle=False
#############################################  데이터 끗  #############################################################

# 모델구성
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):  # 이 클래스를 호출 되었을때 실행 // 통상 이 함수에 들어갈때 레이어의 정의 //  이 모델을 어떻게 정의할것인가?
        # super().__init__()
        super(Model, self).__init__()           #  같이 써줘야함, 아빠다.
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)    # 정의니까 순서 상관 x
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        return
    
    # 순전파!!!
    def forward(self, input_size): # input size : 데이터 
        x = self.linear1(input_size)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x
    
model = Model(30, 1).to(DEVICE)     # (인풋, 아웃풋)


#3 컴파일
criterion = nn.BCELoss()            
optimizer = optim.Adam(model.parameters() , lr=0.001 )  


# model.fit(x,y,epochs = 100 , batch_size = 1)
def train(model , criterion , optimizer, loader):
    total_loss = 0
    
    for x_batch, y_batch in loader :
        
        optimizer.zero_grad()       # zero_grad = optimizer를 0으로 그라디언트(기울기) 초기화 시킨다
        hypothesis = model(x_batch)       # 예상치 값 (순전파)
        loss = criterion(hypothesis , y_batch)    #예상값과 실제값 loss
        
        loss.backward()         # 기울기(gradient)값(loss를 weight로 미분한 값) 계산 -> 역전파 시작
        optimizer.step()        # 가중치 수정(w 갱신) -> 역전파 끝
        # total_loss = total_loss + loss.item()
        total_loss += loss.item()
    return total_loss / len(loader)   # 토탈로스 / 13
    
epochs = 200
for epoch in range(1, epochs + 1) :
    loss = train(model, criterion, optimizer, train_loader)
    print('epoch : {} , loss:{}'.format(epoch,loss))    # verbose

print('==========================================================')

#4. 평가, 예측
# loss = model.evaluate(x,y)
def evaluate(model , criterion , loader) : 
    model.eval()            # 평가모드
    total_loss = 0
    
    y_pred_list = []
    y_true_list = []
    
    for x_batch, y_batch in loader:
        with torch.no_grad():
            y_predict = model(x_batch)
            loss2 = criterion(y_batch , y_predict)
            total_loss += loss2.item()
            y_pred_list.append(y_predict)
            y_true_list.append(y_batch)
    y_pred_list = torch.cat(y_pred_list).cpu().detach().numpy()
    y_true_list = torch.cat(y_true_list).cpu().detach().numpy()
    return total_loss / len(loader), y_pred_list, y_true_list

loss2, y_pred, y_true = evaluate(model , criterion , test_loader)
print('최종 loss : ', loss2)


y_pred = np.round(y_pred)
print(y_pred)
# print(y_test) 
# y_test 결과 : device='cuda:0'
score = accuracy_score(y_true, y_pred)

print('Accuracy: {:.4f}'.format(score)) # Accuracy: 0.9737








