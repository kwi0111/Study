# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터 // 판다스, 넘파이 
path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)    
test_csv = pd.read_csv(path + "test.csv", index_col=0)      # 헤더는 기본 첫번째 줄이 디폴트값
submission_csv = pd.read_csv(path + "submission.csv")       

############## 결측치 처리, 1.제거 ############
train_csv = train_csv.dropna()          # 결측치있으면 행이 삭제됨

############## 결측치 처리, 2.채움 ############
test_csv = test_csv.fillna(test_csv.mean()) 

############ x 와 y를 분리 ################
x = train_csv.drop(['count'], axis=1)       # 행삭제 : axis = 0 // 열삭제 : axis = 1 // train_csv에 있는 'count'열 삭제 
y = train_csv['count']                      # train_csv에 있는 'count'열을 y로 설정


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.9, random_state=123123,
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
test_csv = scaler.transform(test_csv)

#2. 모델구성
model = Sequential()
model.add(Dense(20, input_dim=9))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(27,activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
learning_rate = 0.001
rlr = ReduceLROnPlateau(monitor='val_loss',
                        patience=20,
                        mode='auto',
                        verbose=1,
                        factor=0.5,  # 반으로 줄여라.
                        
                        )
model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))

hist = model.fit(x_train, y_train, epochs=1000, batch_size=10,
          validation_split=0.2,
          callbacks=[rlr],        # 콜백 함수 // 친구들도 더 있다..
          )


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)       # 모델로 예측을 수행하기 위한 함수
y_predict = model.predict(x_test)           # x_test -> y_predict 나옴 (r2값을 위한 예측값)
y_submit = model.predict(test_csv)

r2 = r2_score(y_test, y_predict)
print("lr : {0}, 로스 :{1} ".format(learning_rate, loss))
print("lr : {0}, r2 : {1}".format(learning_rate, r2))


# RMSE :  41.68750689030892
# MSE :  1737.84814453125
# 로스 :  1737.84814453125
# r2 스코어 :  0.7692557406100738


'''
lr : 1.0, 로스 :7612.86083984375 
lr : 1.0, r2 : -0.010803990335116254

lr : 0.1, 로스 :2388.931640625
lr : 0.1, r2 : 0.6828076122456479

lr : 0.01, 로스 :1577.60546875 
lr : 0.01, r2 : 0.7905321104857432

lr : 0.001, 로스 :2023.044921875
lr : 0.001, r2 : 0.7313885203829704
'''

'''rlr 적용
lr : 0.001, 로스 :1805.932861328125
lr : 0.001, r2 : 0.7602157384404008

lr : 0.01, 로스 :1775.285888671875 
lr : 0.01, r2 : 0.7642849025035451

lr : 0.1, 로스 :7564.97021484375 
lr : 0.1, r2 : -0.004445301742466601

lr : 1, 로스 :7552.91552734375
lr : 1, r2 : -0.0028446885922215337
'''

