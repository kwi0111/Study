import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# GPU 를 되는지 안되는지 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ' , torch.__version__ , '사용 DEVICE : ', DEVICE)


#1. 데이터
x = np.array([range(10)])
x = x.transpose()
print(x.shape)                          # (10,1)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
              [9,8,7,6,5,4,3,2,1,0]])

y = y.T                                       

x = torch.FloatTensor(x).to(DEVICE)        # 이걸 해줘야지 torch data로 사용할 수 있음 // unsqueeze(1) 2 차원을 맞춰주는 것
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)        # shape를 똑같이 해줘야 된다 // 똑같지 않으면 그냥 n 빵 때려서 중간값으로 1개가 들어가게 됨 이 때는 2가 들어가게 됨

print(x,y)  # tensor([1., 2., 3.]) tensor([1., 2., 3.])

#2 모델구성
# model = Sequential()
# model.add(Dense(1,input_dim = 1))
# model = nn.Linear(1,1).to(DEVICE) # 인풋 , 아웃풋  # y = xw + b
model = nn.Sequential(
    nn.Linear(1,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,3),
).to(DEVICE)

#3 컴파일
# model.compile(loss = 'mse' , optimizer = 'adam' )
criterion = nn.MSELoss()                # criterion : 표준 -> 이렇게 쓰는 이유는 그냥 제일 많이 써서
# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters() , lr=0.01 )         # SGD = Batch 안에 mini batch를 만들어서 그 안에 있는 것 중 랜덤으로 1개를 지정해서 훈련하고 나머지는 버림
                                                            # 연산량이 빨라지고 성능이 좋을 수 도 있음 , 다른 에포에서는 다른 것을 쓸 수 있어서 데이터를 100% 버린다고는 할 수 없음

# model.fit(x,y,epochs = 100 , batch_size = 1)
def train(model , criterion , optimizer, x, y):
    
    ####### 순전파 #######
    # model.train()     # 훈련모드 , 디폴트 값 // 훈련모드랑 dropout , batch normalization 같은것을 사용
    # w = w - lr * (loss를 weight로 미분한 값)
    optimizer.zero_grad()       # zero_grad = optimizer를 0으로 초기화 시킨다
                                # 1. 처음에 0으로 시작하는게 좋아서
                                # 2. epoch가 돌때마다 전의 gradient를 가지고 있어서 그게 문제가 될 수 있어서 이걸 해결 하기 위해서
                                #    계속 0으로 바꿔주는 것이다. 

    hypothesis = model(x)       # 예상치 값 (순전파)
    
    loss = criterion(hypothesis , y)    #예상값과 실제값 loss
    
    #####################
    
    loss.backward()         # 기울기(gradient)값(loss를 weight로 미분한 값) 계산 -> 역전파 시작
    optimizer.step()        # 가중치 수정(w 갱신)       -> 역전파 끝
    return loss.item()      # item을 쓰는 이유는 numpy 데이터로 뽑기위해서 똑같이 tensor 데이터는 맞음
    
epochs = 1000
for epoch in range(1 , epochs+ 1 ) :
    loss = train(model , criterion, optimizer , x , y)
    print('epoch : {} , loss:{}'.format(epoch,loss))    # verbose

print('==========================================================')

#4. 평가, 예측
# loss = model.evaluate(x,y)
def evaluate(model , criterion , x,y ) : 
    model.eval()            # 평가모드 , 안해주면 평가가 안됨 dropout 같은 것들이 들어가게 됨
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y , y_predict)
    return loss2.item()

loss2 = evaluate(model , criterion , x, y)
print('최종 loss : ', loss2 )

# result = model.predict([4])
result = model(torch.Tensor([[10]]).to(DEVICE))
print('4의 예측값은 :' , result.tolist() )


# 최종 loss :  0.00011766229727072641
# 4의 예측값은 : 4.021754264831543