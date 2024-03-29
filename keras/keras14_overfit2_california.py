import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터
datasets = fetch_california_housing()                               # fetch : 가져옴
x = datasets.data                                                   # 샘플 데이터
y = datasets.target                                                 # 라벨 데이터

print(x)
print(y)
print(x.shape, y.shape)                                             #(20640, 8) (20640,) 인풋8 아웃풋1

print(datasets.feature_names)                                       # feature 데이터의 이름 
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)                                               # describe의 약자로 데이터에 대한 설명

x_train, x_test, y_train, y_test = train_test_split(x, y,                     # 훈련 데이터, 테스트 데이터 나누는 과정
                                                    train_size=0.7,
                                                    random_state=123,     
                                                    shuffle=True,
                                                    )

#2. 모델 구성 
model = Sequential()                                                            # 순차적으로 레이어 층을 더해서 만든다.
model.add(Dense(10, input_dim=8))                                               # 입력 노드 8, 출력 노드 10
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))                                                             # 출력 노드 1

#3. 컴파일, 훈련 
model.compile(loss="mse", optimizer='adam')                                     # model.compile: 학습에 필요한것을 번역 // mse : 평균 제곱 오차 // optimizer : 훈련 과정
start_time = time.time()                                                        # 현재 시간
hist = model.fit(x_train, y_train, epochs=100, batch_size=250, 
          validation_split=0.3,
          verbose=1
          )                       # 모델 학습
end_time = time.time()                                                          # 끝나는 시간

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                                           # 평가는 항상 테스트 데이터
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score  
r2 = r2_score(y_test, y_predict)                                                # 실제값, 예측값 순서
print("로스 : ", loss)
print("r2 스코어 : " , r2)
print("걸린시간 : ", round(end_time - start_time, 2),"초")     

def RMSE(aaa, bbb):
    return np.sqrt(mean_squared_error(aaa, bbb))
rmse = RMSE(y_test, y_predict)
print("RMSE : " , rmse)
print("MSE : ", loss)

print("걸린시간 : ", round(end_time - start_time, 2),"초")

print("==========================")
print(hist)
print("============= hist.history =============")
print(hist.history)         # 딕셔너리 {} : 키(로스,loss), 벨류(숫자,값) 한쌍 //
                            # 리스트 []: 두개이상
print("============ loss ============")
print(hist.history['loss'])
print("=========== val_loss ==========")
print(hist.history['val_loss'])
print("===============================")

import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'    # 위치

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss', marker='.')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')
plt.legend(loc='upper right') # 라벨
plt.title('캘리포니아 LOSS') #제목
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()


