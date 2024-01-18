# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time

#1. 데이터 // 판다스, 넘파이 
path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)    
print(train_csv)                                            # 여기 train_csv에서 훈련/테스트 데이터 나눠야함.
test_csv = pd.read_csv(path + "test.csv", index_col=0)      # 헤더는 기본 첫번째 줄이 디폴트값
print(test_csv)                                             # 위의 훈련 데이터로 예측해서 count값 찾아야함. (문제집)
submission_csv = pd.read_csv(path + "submission.csv")       
print(submission_csv)                                       # 서브미션 형식 그대로 제출해야함.

############## 결측치 처리, 1.제거 ############
# print(train_csv.isnull().sum())       
print(train_csv.isna().sum())           # 데이터 프레임 결측치 확인
train_csv = train_csv.dropna()          # 결측치있으면 행이 삭제됨
print(train_csv.isna().sum())           # train 결측치 삭제 후 확인
print(train_csv.info())
print(train_csv.shape)                  # (1328, 10)        // 1459 - 1328 = 121열 삭제

############## 결측치 처리, 2.채움 ############
test_csv = test_csv.fillna(test_csv.mean())     # test 결측치를 평균인 중간값으로 채움.
print(test_csv.info())

############ x 와 y를 분리 ################
x = train_csv.drop(['count'], axis=1)       # 행삭제 : axis = 0 // 열삭제 : axis = 1 // train_csv에 있는 'count'열 삭제 
print(x)
y = train_csv['count']                      # train_csv에 있는 'count'열을 y로 설정
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.9, random_state=123123,
)
print(x_train.shape, x_test.shape)      # (1195, 9) (133, 9)
print(y_train.shape, y_test.shape)      # (1195,) (133,)        train_size 달라지면 바뀜
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# scaler = MinMaxScaler() # 클래스 정의
scaler = StandardScaler() # 클래스 정의
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
model.add(Dense(200, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(270,activation='relu'))
model.add(Dense(180, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# es = EarlyStopping(monitor='val_loss',      # EarlyStopping의 기준이 되는 값
#                    mode='min',          # auto, min, max
#                    patience=300,
#                    verbose=1,
#                    restore_best_weights=True        # 최고의 저장 가중치를 불러와라. // 디폴트 : False // 트레인에서 실행 // 테스트에서는 안쓰임
#                    )
mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath='../_data/_save/MCP/keras26_ddarung_MCP1.hdf5'
                      )
start_time = time.time()   #현재 시간
hist = model.fit(x_train, y_train, epochs=1000, batch_size=10,
          validation_split=0.2,
          callbacks=[mcp],        # 콜백 함수 // 친구들도 더 있다..
          )
end_time = time.time()   #끝나는 시간

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)       # 모델로 예측을 수행하기 위한 함수
y_predict = model.predict(x_test)           # x_test -> y_predict 나옴 (r2값을 위한 예측값)
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)       # (715, 1)

######### submission.csv 만들기 ('count'칼럼에 값만 넣어주면 됨) #############
submission_csv['count'] = y_submit  # y_submit = model.predict(test_csv) (715, 1) 에서 나온 데이터를 count행에 넣어줌.
print(submission_csv)
print(submission_csv.shape) # (715, 2) // id, count


# submission_csv.to_csv(path + "submission_0110.csv", index=False)        #  to_csv 혹은 to_excel 함수를 사용할 때 'index=False' 추가


r2 = r2_score(y_test, y_predict)        # 회귀 모델의 성능에 대한 평가지표 0 < r2 < 1
                              
def RMSE(aaa, bbb):
    return np.sqrt(mean_squared_error(aaa, bbb))
rmse = RMSE(y_test, y_predict)
print("RMSE : " , rmse)
print("MSE : ", loss)
print("로스 : ", loss)
print("r2 스코어 : " , r2)
print("걸린시간 : ", round(end_time - start_time, 2),"초")

# 씨피유
# RMSE :  40.69221742892787
# MSE :  1655.8565673828125
# 로스 :  1655.8565673828125
# r2 스코어 :  0.7801422536953404
# 걸린시간 :  124.08 초
