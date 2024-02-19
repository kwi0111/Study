import numpy as np
import pandas as pd #판다스에 데이터는 넘파이 형태로 들어가있음.
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import time
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV

#1. 데이터
path = "C:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")
print(train_csv.columns)
# ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#       'humidity', 'windspeed', 'casual', 'registered', 'count']
x = train_csv.drop(['casual','registered','count'], axis=1)
y = train_csv['count']

from sklearn.model_selection import train_test_split,KFold,cross_val_score, GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(
                                                    x,
                                                    y,             
                                                    train_size=0.86,
                                                    random_state=2024,
                                                    shuffle=True,
                                                    # stratify=y,
                                                    )
n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123123)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123123)

scaler = MinMaxScaler()
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

# 4. 평가 예측
print('====================================')
print("최상의 매개변수 : ", model.best_estimator_) 
print("최상의 매개변수 : ", model.best_params_) 
print("최상의 점수 : ", model.best_score_)  
results = model.score(x_test, y_test)
print("최고의 점수 : ", results)  
print("걸린시간 :", round(end_time - start_time, 2), "초")

# 최상의 매개변수 :  {'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0.1, 'n_estimators': 500, 'min_child_weight': 1, 'max_depth': 7, 'learning_rate': 0.01, 'gamma': 3, 'colsample_bytree': 0.5}
# 최상의 점수 :  0.350204414133509
# 최고의 점수 :  0.3498393132471771
# 걸린시간 : 6.74 초
