#https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV


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

print(x.info())

x_train, x_test, y_train, y_test = train_test_split(
                                                    x,
                                                    y,             
                                                    train_size=0.86,
                                                    random_state=2024,
                                                    shuffle=True,
                                                    # stratify=y,
                                                    )
print(x_train.info())
# train 9개중에서 7개로 만들었으니까 test도 7개로 만들어 줘야함
#  0   hour                    1167 non-null   int64
#  1   hour_bef_temperature    1167 non-null   float64
#  2   hour_bef_precipitation  1167 non-null   float64
#  3   hour_bef_visibility     1167 non-null   float64
#  4   hour_bef_ozone          1167 non-null   float64
#  5   hour_bef_pm10           1167 non-null   float64
#  6   hour_bef_pm2.5          1167 non-null   float64

#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1459 non-null   float64
#  2   hour_bef_precipitation  1459 non-null   float64
#  3   hour_bef_windspeed      1459 non-null   float64
#  4   hour_bef_humidity       1459 non-null   float64
#  5   hour_bef_visibility     1459 non-null   float64
#  6   hour_bef_ozone          1459 non-null   float64
#  7   hour_bef_pm10           1459 non-null   float64
#  8   hour_bef_pm2.5          1459 non-null   float64

n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123123)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123123)

# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from xgboost import XGBRegressor
parameters = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.001, 0.0001],
    'max_depth': [3, 5, 7, 9, 11],
    'min_child_weight': [0, 0.01, 0.1, 1, 5, 10, 100],
    'subsample': [0.5, 0.7, 0.9, 1],
    'colsample_bytree': [0.5, 0.7, 0.9, 1],
    'gamma': [0, 1, 2, 3, 4, 5],
    'reg_alpha': [0, 0.01, 0.1, 1, 10],
    'reg_lambda': [0, 0.01, 0.1, 1, 10]
}
xgb = XGBRegressor(random_state=123)
model = RandomizedSearchCV(xgb, parameters, cv=kfold, random_state=123123,
                           n_jobs=22)

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

from sklearn.metrics import r2_score
print('r2_score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
y_submit = model.predict(test_csv)

# 4. 평가 예측
print('====================================')
print("최상의 매개변수 : ", model.best_estimator_) 
print("최상의 매개변수 : ", model.best_params_) 
print("최상의 점수 : ", model.best_score_)  
results = model.score(x_test, y_test)
print("최고의 점수 : ", results)  
print("걸린시간 :", round(end_time - start_time, 2), "초")

# ####### submission.csv 만들기 (count컬럼에 값만 넣어주면 됨) #####
# submission_csv['count'] = y_submit
# print(submission_csv)


# path = "c:\\_data\\dacon\\ddarung\\"
# import time as tm
# ltm = tm.localtime(tm.time())
# save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
# submission_csv.to_csv(path + f"submission_{save_time}{rmse:.3f}.csv", index=False)


# 최상의 매개변수 :  {'subsample': 0.7, 'reg_lambda': 0, 'reg_alpha': 0.01, 'n_estimators': 400, 'min_child_weight': 0.1, 'max_depth': 11, 'learning_rate': 0.01, 'gamma': 4, 'colsample_bytree': 1}        
# 최상의 점수 :  0.8029933483333599
# 최고의 점수 :  0.7561692934655944
# 걸린시간 : 6.86 초

