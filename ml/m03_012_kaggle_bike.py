# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.svm import LinearSVR
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import time

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
print(x_train.shape, x_test.shape)     # (7620, 8) (3266, 8)        train : 교과서 , test : 모의고사
print(y_train.shape, y_test.shape)     # (7620,) (3266,)
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
models = [LinearSVR(),LinearRegression(),KNeighborsRegressor(),DecisionTreeRegressor(),RandomForestRegressor()]
############## 훈련 반복 for 문 ###################
for model in models :
    try:
        model.fit(x_train, y_train)
        result = model.score(x_test, y_test)
        print(f'{type(model).__name__} score : ', round(result, 2))
        
        y_predict = model.predict(x_test)
        print(f'{type(model).__name__} predict : ', round(r2_score(y_test,y_predict), 2))
    except:
        continue

'''
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
print("r2 스코어 : " , r2)           # (테스트 데이터, 예측 데이터)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


# LinearSVR                   0.08893457626110868
# LinearRegression            0.2375332998296259
# KNeighborsRegressor         0.1657977079275006
# DecisionTreeRegressor       -0.1509105931507726
# RandomForestRegressor       0.2499779602479224
'''
'''
LinearSVR score :  0.2
LinearSVR predict :  0.2
LinearRegression score :  0.24
LinearRegression predict :  0.24
KNeighborsRegressor score :  0.24
KNeighborsRegressor predict :  0.24
DecisionTreeRegressor score :  -0.15
DecisionTreeRegressor predict :  -0.15
RandomForestRegressor score :  0.24
RandomForestRegressor predict :  0.24
'''

