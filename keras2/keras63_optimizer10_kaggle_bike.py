# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터 (분석, 정제, 전처리) // 판다스 
path = "C:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)      # 날짜 데이터 인덱스로
submission_csv = pd.read_csv(path + "samplesubmission.csv")

# 결측치 처리
print(train_csv.isna().sum())           # 데이터 프레임 결측치  없다.

# x와 y를 분리
x = train_csv.drop(['casual','registered','count'], axis=1)       # 행삭제 : axis = 0 // 열삭제 : axis = 1 // train_csv에 있는 'count'열 삭제 
y = train_csv['count']                      # train_csv에 있는 'count'열을 y로 설정

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    train_size=0.9,
                                                    random_state=123,
                                                    shuffle=True
                                                    )

#2. 모델구성
model = Sequential()
model.add(Dense(20, input_dim=8, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(20, activation='relu'))     # 활성화 함수 'Relu' : 0이하는 0 / 0이상은 그 값  // 음수나 튀는 데이터 렐루로 감싼다. //
model.add(Dense(1, activation='relu'))                         # 여기까지 렐루 쓸지 안쓸지 생각 // 안써있으면 디폴트 linear(선형)

#3. 컴파일, 훈련
from keras.optimizers import Adam
learning_rate = 0.00001
model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))

es = EarlyStopping(monitor='val_loss',      # EarlyStopping의 기준이 되는 값
                   mode='min',          # auto, min, max
                   patience=15,
                   verbose=1,
                   restore_best_weights=True        # 최고의 저장 가중치를 불러와라. // 디폴트 : False // 트레인에서 실행 // 테스트에서는 안쓰임
                   )

hist = model.fit(x_train, y_train, epochs=200, batch_size=10,
          validation_split=0.2,
          callbacks=[es],        # 콜백 함수 // 친구들도 더 있다..
          )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)       # 모델로 예측을 수행하기 위한 함수
y_predict = model.predict(x_test)           # x_test -> y_predict 나옴 (r2값을 위한 예측값)

r2 = r2_score(y_test, y_predict)
print("lr : {0}, 로스 :{1} ".format(learning_rate, loss))
print("lr : {0}, r2 : {1}".format(learning_rate, r2))

# 그냥
# MSE :  68482.890625

# MinMaxScaler
# MSE :  22144.33203125

# StandardScaler
# MSE :  21989.283203125

# MaxAbsScaler
# MSE :  22550.693359375

# RobustScaler
# MSE :  22350.0703125

'''
lr : 1e-05, 로스 :24838.435546875
lr : 1e-05, r2 : 0.21673057275780794

lr : 0.001, 로스 :22442.755859375
lr : 0.001, r2 : 0.29227716116013447

lr : 0.01, 로스 :23779.01171875 
lr : 0.01, r2 : 0.25013893361181727

lr : 0.1, 로스 :68482.890625
lr : 0.1, r2 : -1.1595790199052693

lr : 1.0, 로스 :68482.890625
lr : 1.0, r2 : -1.1595790199052693
'''



