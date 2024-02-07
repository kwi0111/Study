# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
from sklearn.svm import LinearSVR
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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
# scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
scaler = RobustScaler() # 클래스 정의


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


#2. 모델구성
# model = LinearSVR(C=100)
# model = LinearRegression()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()

#3. 컴파일, 훈련
start_time = time.time()   #현재 시간
model.fit(x_train, y_train)
end_time = time.time()   #끝나는 시간

#4. 평가, 예측
results = model.score(x_test, y_test) 
y_predict = model.predict(x_test) 
print("acc : ", results)
print("걸린시간 : ", round(end_time - start_time, 2),"초")  
from sklearn.metrics import r2_score  
r2 = r2_score(y_test, y_predict)                                                # 실제값, 예측값 순서
print("r2 스코어 : " , r2)       # 회귀 모델의 성능에 대한 평가지표 0 < r2 < 1
                              
def RMSE(aaa, bbb):
    return np.sqrt(mean_squared_error(aaa, bbb))
rmse = RMSE(y_test, y_predict)
print("RMSE : " , rmse)


# LinearSVR                   0.6322177787256706
# LinearRegression            0.6726297445668414
# KNeighborsRegressor         0.7257135057206954
# DecisionTreeRegressor       0.7524878637597527
# RandomForestRegressor       0.8367343854943599