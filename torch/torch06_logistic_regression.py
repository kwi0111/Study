import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# GPU 를 되는지 안되는지 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ' , torch.__version__ , '사용 DEVICE : ', DEVICE)


#1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(x.shape, y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape)

#2 모델구성
# model = Sequential()
# model.add(Dense(1,input_dim = 1))
# model = nn.Linear(1,1).to(DEVICE) # 인풋 , 아웃풋  # y = xw + b
model = nn.Sequential(
    nn.Linear(30, 64),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.Linear(16, 7),
    nn.Linear(7, 1),
    nn.Sigmoid(),
).to(DEVICE)

#3 컴파일
criterion = nn.BCELoss()            
optimizer = optim.Adam(model.parameters() , lr=0.001 )  


# model.fit(x,y,epochs = 100 , batch_size = 1)
def train(model , criterion , optimizer, x_train, y_train):
    
    ####### 순전파 #######
    # model.train()     # 훈련모드 , 디폴트 값 // 훈련모드랑 dropout , batch normalization 같은것을 사용
    # w = w - lr * (loss를 weight로 미분한 값)
    optimizer.zero_grad()       # zero_grad = optimizer를 0으로 초기화 시킨다
                                # 1. 처음에 0으로 시작하는게 좋아서
                                # 2. epoch가 돌때마다 전의 gradient를 가지고 있어서 그게 문제가 될 수 있어서 이걸 해결 하기 위해서
                                #    계속 0으로 바꿔주는 것이다. 

    hypothesis = model(x_train)       # 예상치 값 (순전파)
    
    loss = criterion(hypothesis , y_train)    #예상값과 실제값 loss
    
    #####################
    
    loss.backward()         # 기울기(gradient)값(loss를 weight로 미분한 값) 계산 -> 역전파 시작
    optimizer.step()        # 가중치 수정(w 갱신)       -> 역전파 끝
    return loss.item()      # item을 쓰는 이유는 numpy 데이터로 뽑기위해서 똑같이 tensor 데이터는 맞음
    
epochs = 2000
for epoch in range(1 , epochs+ 1 ) :
    loss = train(model , criterion, optimizer , x_train, y_train)
    print('epoch : {} , loss:{}'.format(epoch,loss))    # verbose

print('==========================================================')

#4. 평가, 예측
# loss = model.evaluate(x,y)
def evaluate(model , criterion , x_test, y_test) : 
    model.eval()            # 평가모드 , 안해주면 평가가 안됨 dropout 같은 것들이 들어가게 됨
    
    with torch.no_grad():
        y_predict = model(x_test)
        loss2 = criterion(y_test , y_predict)
    return loss2.item(), y_predict

loss2, y_predict = evaluate(model , criterion , x_test, y_test)
print('최종 loss : ', loss2 )

# 예측 결과 이진 분류로 변환
y_predict = (y_predict > 0.5).float()
y_test = y_test.cpu().numpy()
y_predict = y_predict.cpu().numpy()

# 성능 평가
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')




