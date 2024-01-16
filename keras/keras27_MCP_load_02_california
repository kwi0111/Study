import numpy as np                                                  # numpy 빠른 계산을 위해 지원되는 파이썬 라이브러리
import time
from keras.models import Sequential, load_model
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
print(datasets.DESCR)                                               # describe의 약자로 데이터에 대한 설명

x_train, x_test, y_train, y_test = train_test_split(x, y,                     # 훈련 데이터, 테스트 데이터 나누는 과정
                                                    train_size=0.7,
                                                    random_state=123,     
                                                    shuffle=True,
                                                    )
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
scaler = MinMaxScaler() # 클래스 정의
# scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델 구성 
# model = Sequential()                                                            # 순차적으로 레이어 층을 더해서 만든다.
# model.add(Dense(10, input_dim=8, activation='relu'))                                               # 입력 노드 8, 출력 노드 10
# model.add(Dense(40, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(1))                                                             # 출력 노드 1

# #3. 컴파일, 훈련 
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss',
#                    mode='auto',
#                    patience=300,
#                    verbose=1,
#                    restore_best_weights=True
#                    )
# mcp = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       verbose=1,
#                       save_best_only=True,
#                       filepath='../_data/_save/MCP/keras26_california_MCP1.hdf5'
#                       )
# model.compile(loss="mse", optimizer='adam')                                     
# model.fit(x_train, y_train, epochs=10000, batch_size=250, 
#           validation_split=0.18,
#           verbose=1,
#           callbacks=[es, mcp]
#           )                   
model = load_model('../_data/_save/MCP/keras26_california_MCP1.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                                       
y_predict = model.predict(x_test)
results = model.predict(x)

from sklearn.metrics import r2_score  
r2 = r2_score(y_test, y_predict)                                                
print("로스 : ", loss)
print("r2 스코어 : " , r2)

