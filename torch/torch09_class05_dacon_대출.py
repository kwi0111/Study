# 1. 캔서 (이진)
# 2. digits
# 3. fetch_covtype
# 4. dacon_wine
# 5. dacon_대출
# 6. kaggle_비만도

# 7. load_diabetes
# 8. california
# 9. dacon 따릉이
# 10. kaggle bike

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# GPU 를 되는지 안되는지 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ' , torch.__version__ , '사용 DEVICE : ', DEVICE)


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
# y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
# y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
y_train = torch.LongTensor(y_train.values).to(DEVICE)
y_test = torch.LongTensor(y_test.values).to(DEVICE)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# 클래스로 바꿀거임
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):  # 이 클래스를 호출 되었을때 실행 // 통상 이 함수에 들어갈때 레이어의 정의 //  이 모델을 어떻게 정의할것인가?
        # super().__init__()
        super(Model, self).__init__()           #  같이 써줘야함, 아빠다.
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)    # 정의니까 순서 상관 x
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
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
        return x
    
model = Model(12, 7).to(DEVICE)     # (인풋, 아웃풋)


#3 컴파일
criterion = nn.CrossEntropyLoss()            
optimizer = optim.Adam(model.parameters() , lr=0.001)  


# model.fit(x,y,epochs = 100 , batch_size = 1)
def train(model , criterion , optimizer, x, y):
    optimizer.zero_grad()     
    hypothesis = model(x)       # 예상치 값 (순전파)
    loss = criterion(hypothesis , y)    #예상값과 실제값 loss
    loss.backward()         # 기울기(gradient)값(loss를 weight로 미분한 값) 계산 -> 역전파 시작
    optimizer.step()        # 가중치 수정(w 갱신)       -> 역전파 끝
    return loss.item()      # item을 쓰는 이유는 numpy 데이터로 뽑기위해서 똑같이 tensor 데이터는 맞음
    
epochs = 2000
for epoch in range(1, epochs + 1) :
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch : {} , loss:{}'.format(epoch,loss))    # verbose

print('==========================================================')


#4. 평가, 예측
# loss = model.evaluate(x,y)
def evaluate(model , criterion , x_test, y_test) : 
    model.eval()            # 평가모드 , 안해주면 평가가 안됨 dropout 같은 것들이 들어가게 됨
    
    with torch.no_grad():
        y_predict = model(x_test)
        print(y_test.shape, y_predict.shape)
        loss2 = criterion(y_predict, y_test)
    return loss2.item()

loss2 = evaluate(model , criterion , x_test, y_test)
print('최종 loss : ', loss2)

# y_predict = model(x_test)
# 위의 결과 : device='cuda:0', grad_fn=<SigmoidBackward0>) 

y_predict = torch.argmax(model(x_test) ,dim=1).cpu().detach().numpy()
print(y_predict)
# print(y_test) 
# y_test 결과 : device='cuda:0'
score = accuracy_score(y_test.cpu().numpy(), y_predict)

print('Accuracy: {:.4f}'.format(score)) # Accuracy: 0.5227






