# 랜덤포레스트의 알고리즘이 배깅이다.
# 배깅 - 보팅
# votiong - 모델 여러개 / 같은 데이터 / 소프트 or 하드 / 하드는 다수결 / 소프트는 평균에서 높은쪽
# bagging - 모델 1개 - 데이터가 다르다 (샘플링해서) // 에포 느낌 = n이스터메이트 //  중복 된다 // 

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LogisticRegression # 앤 분류다

#1. 데이터
path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0) # \ \\ / // 다 가능( 예약어 사용할때 두개씩 사용) 인덱스컬럼은 0번째 컬럼이다라는뜻.
test_csv = pd.read_csv(path +"test.csv", index_col=0)
submission_csv = pd.read_csv(path + "submission.csv") 

print(train_csv.info())
train_csv = train_csv.fillna(train_csv.mean())  #결측치가 하나라도 있으면 행전체 삭제됨.
test_csv = test_csv.fillna(test_csv.mean())   # (0,mean)

# test_csv = test_csv.drop(['hour_bef_humidity','hour_bef_windspeed'], axis=1)   # (0,mean)

print(train_csv.shape)      #(1328, 10)

################# x와 y를 분리 ###########
x = train_csv.drop(['count',], axis=1)
y = train_csv['count']


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size= 0.8)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
model = BaggingRegressor(RandomForestRegressor(),
                          n_estimators=100,
                          n_jobs=20,
                          random_state=123,
                        #   bootstrap=True,   # 디폴트다 // 중복 허용 // False = 중복 허용 X
                          bootstrap=False,   # 디폴트다 // 중복 허용 // False = 중복 허용 X
                          
                          )   # 랩핑

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수', results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2_score :', r2)

##########################
# r2_score : 0.7341018017406806 -> n_estimators=100으로 r2_score : 0.7381181281326573
# 100, False 최종점수 0.7514021768862288

'''



'''
















