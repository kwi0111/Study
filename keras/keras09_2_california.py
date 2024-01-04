import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# import warnings
# warnings.filterwarnings('ignore')


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(20640, 8) (20640,) 인풋8 아웃풋1

print(datasets.feature_names)       
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,               
                                                    train_size=0.9,
                                                    random_state=0,     
                                                    shuffle=True,
                                                    )

#2. 모델 구성 
model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss="mse", optimizer='adam')
start_time = time.time()   #현재 시간
model.fit(x_train, y_train, epochs=10000, batch_size=250)
end_time = time.time()   #끝나는 시간
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)   # 평가는 항상 테스트 데이터
y_predict = model.predict(x_test)
results = model.predict(x)

from sklearn.metrics import r2_score  
r2 = r2_score(y_test, y_predict)        # 실제값, 예측값 순서
print("로스 : ", loss)
print("r2 스코어 : " , r2)
print("걸린시간 : ", round(end_time - start_time, 2),"초")     

# [실습] 만드시오.
# R2 0.55 ~ 0.6 이상

# 트레인 사이즈 높이니까 좋아짐 // 에포와 배치 사이즈 조절 3000
# 로스 :  0.5571222901344299
# r2 스코어 :  0.5946174290684143
# 걸린시간 :  133.34 초

# 에포 10000
# 로스 :  0.5398460030555725
# r2 스코어 :  0.6071883105238739
# 걸린시간 :  449.28 초


