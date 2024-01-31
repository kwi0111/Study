import numpy as np                                                  # numpy 빠른 계산을 위해 지원되는 파이썬 라이브러리
import time
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, Conv1D
from keras.callbacks import EarlyStopping
from sklearn.datasets import fetch_california_housing               # 사이킷런 : 파이썬 머신러닝 라이브러리 // sklearn에서 제공하는 데이터셋
from sklearn.model_selection import train_test_split                # scikit-learn 패키지 중 model_selection에서 데이터 분할
# import warnings                                                   # 터미널 경고 무시
# warnings.filterwarnings('ignore')


#1. 데이터
datasets = fetch_california_housing()                               # fetch : 가져옴
x = datasets.data                                                   # 샘플 데이터
y = datasets.target                                                 # 라벨 데이터

print(x.shape, y.shape)                                             #(20640, 8) (20640,) 인풋8 아웃풋1
print(datasets.feature_names)                                       # feature 데이터의 이름 
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)                                               # describe의 약자로 데이터에 대한 설명

x_train, x_test, y_train, y_test = train_test_split(x, y,                     # 훈련 데이터, 테스트 데이터 나누는 과정
                                                    train_size=0.8,
                                                    random_state=123,     
                                                    shuffle=True,
                                                    )
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# scaler = MinMaxScaler() # 클래스 정의
# scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
scaler = RobustScaler() # 클래스 정의



x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train = x_train.reshape(-1, 8, 1)
x_test = x_test.reshape(-1, 8, 1)

print(x_train.shape, x_test.shape)  # (16512, 8, 1) (4128, 8, 1)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(7, 2, padding='same', input_shape = (8,1)))
model.add(Conv1D(10, 2, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))                                                       

#3. 컴파일, 훈련 
model.compile(loss="mse", optimizer='adam')                                    
es = EarlyStopping(monitor='loss',
                   mode='auto',
                   patience=15,
                   verbose=1,
                   restore_best_weights=True,
                   )
model.fit(x_train, y_train, epochs=200, batch_size=250, 
          validation_split=0.2,
          verbose=1
          )                  

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
y_predict = model.predict(x_test)
# results = model.predict(x)
print("로스 : ", loss)
                                         


'''
# 그냥
# 로스 :  1.5651301145553589

# MinMaxScaler
# 로스 :  0.5345919728279114

# StandardScaler
# 로스 :  0.5433641672134399

# MaxAbsScaler
# 로스 :  0.6419296860694885

# RobustScaler
# 로스 :  0.5240345001220703

CNN

LSTM
로스 :  0.3573364317417145

Conv1D
로스 :  0.2770712077617645


'''



