# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
import time
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error

#1. 데이터 (분석, 정제, 전처리) // 판다스 
path = "C:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)
print(train_csv.shape) # (10886, 11)
print(train_csv.info())   


test_csv = pd.read_csv(path + "test.csv", index_col=0)      # 날짜 데이터 인덱스로
print(test_csv) 
print(test_csv.shape) # (6493, 8)
print(test_csv.info())  


submission_csv = pd.read_csv(path + "samplesubmission.csv")
print(submission_csv)
print(submission_csv.shape)  # (6493, 2)

# 결측치 처리
print(train_csv.isna().sum())           # 데이터 프레임 결측치  없다.


# x와 y를 분리
x = train_csv.drop(['casual','registered','count'], axis=1)       # 행삭제 : axis = 0 // 열삭제 : axis = 1 // train_csv에 있는 'count'열 삭제 
print(x)

y = train_csv['count']                      # train_csv에 있는 'count'열을 y로 설정
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    train_size=0.9,
                                                    random_state=123,
                                                    shuffle=True
                                                    )
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# scaler = MinMaxScaler() # 클래스 정의
scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)     # (7620, 8) (3266, 8)        train : 교과서 , test : 모의고사
print(y_train.shape, y_test.shape)     # (7620,) (3266,)

#2. 모델구성
model = Sequential()
model.add(Dense(20, input_dim=8, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(20, activation='relu'))     # 활성화 함수 'Relu' : 0이하는 0 / 0이상은 그 값  // 음수나 튀는 데이터 렐루로 감싼다. //
model.add(Dense(1, activation='relu'))                         # 여기까지 렐루 쓸지 안쓸지 생각 // 안써있으면 디폴트 linear(선형)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss',
#                    mode='min',
#                    patience=200,
#                    verbose=1,
#                    restore_best_weights=True
#                    )
mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath='../_data/_save/MCP/keras26_bike_MCP1.hdf5'
                      )

start_time = time.time()   #현재 시간
hist = model.fit(x_train, y_train, epochs=1040, batch_size=30, verbose=2,
          validation_split=0.3,
          callbacks=[mcp]
          )
end_time = time.time()   #끝나는 시간

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)       # 모델로 예측을 수행하기 위한 함수
y_predict = model.predict(x_test)           # x_test를 넣으면 y_predict 나옴 // r2를 위한 예측값
y_submit = model.predict(test_csv)          # test_csv : 문제집 //submission : 수능

print(y_submit)
print(y_submit.shape)   #(6493, 1)
submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape) # (6493, 2)

submission_csv.to_csv(path + "submission_0110.csv", index=False) 


r2 = r2_score(y_test, y_predict)            # (테스트 데이터, 예측 데이터)

############################################
# print(submission_csv[submission_csv['count']>0])
print("음수 갯수 : ", submission_csv[submission_csv['count']<0].count()) ## 암기 // 데이터 프레임의 조건 

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
print("r2 스코어 : " , r2)
print("MSE : ", loss)
print("걸린시간 : ", round(end_time - start_time, 2),"초")

# 씨피유
# RMSE :  157.7271813932348
# r2 스코어 :  0.2154871635408856
# MSE :  24877.86328125
# 걸린시간 :  168.64 초



