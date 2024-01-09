import numpy as np                                                  # numpy 빠른 계산을 위해 지원되는 파이썬 라이브러리
import time
from keras.models import Sequential
from keras.layers import Dense

from sklearn.datasets import fetch_california_housing               # 사이킷런 : 파이썬 머신러닝 라이브러리 // sklearn에서 제공하는 데이터셋
from sklearn.model_selection import train_test_split                # scikit-learn 패키지 중 model_selection에서 데이터 분할
# import warnings                                                   # 터미널 경고 무시
# warnings.filterwarnings('ignore')


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
model.fit(x_train, y_train, epochs=1000, batch_size=250, 
          validation_split=0.5,
          verbose=0
          )                       # 모델 학습
end_time = time.time()                                                          # 끝나는 시간

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                                           # 평가는 항상 테스트 데이터
y_predict = model.predict(x_test)
results = model.predict(x)

from sklearn.metrics import r2_score  
r2 = r2_score(y_test, y_predict)                                                # 실제값, 예측값 순서
print("로스 : ", loss)
print("r2 스코어 : " , r2)
print("걸린시간 : ", round(end_time - start_time, 2),"초")                       # round(끝나는 시간 - 현재 시간, 소수점 2째자리까지 나타남)

# 로스 :  0.6051582098007202
# r2 스코어 :  0.5423404260199446
# 걸린시간 :  35.64 초

# validation_split = 0.3
# 로스 :  0.6658098697662354
# r2 스코어 :  0.49647183834467457
# 걸린시간 :  47.41 초

# 0.5
# 로스 :  0.6557255983352661
# r2 스코어 :  0.5040982130158767
# 걸린시간 :  40.99 초

