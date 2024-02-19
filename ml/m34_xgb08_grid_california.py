import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time
from sklearn.svm import LinearSVR
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV
import pandas as pd

#1. 데이터
datasets = fetch_california_housing()
x = np.array(datasets.data)
y = np.array(datasets.target)

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

# 최상의 매개변수 :  {'subsample': 0.7, 'reg_lambda': 0, 'reg_alpha': 0.01, 'n_estimators': 400, 'min_child_weight': 0.1, 'max_depth': 11, 'learning_rate': 0.01, 'gamma': 4, 'colsample_bytree': 1}        
# 최상의 점수 :  0.8129069425930741
# 최고의 점수 :  0.8147210876628694
# 걸린시간 : 10.4 초
