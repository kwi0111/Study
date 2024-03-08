# 랜덤포레스트의 알고리즘이 배깅이다.
# 배깅 - 보팅
# votiong - 모델 여러개 / 같은 데이터 / 소프트 or 하드 / 하드는 다수결 / 소프트는 평균에서 높은쪽
# bagging - 모델 1개 - 데이터가 다르다 (샘플링해서) // 에포 느낌 = n이스터메이트 //  중복 된다 // 

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression # 앤 분류다
from sklearn.ensemble import StackingRegressor
from catboost import CatBoostRegressor
from bayes_opt import BayesianOptimization
import time

datasets = load_diabetes()


x = datasets['data']
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size= 0.8)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
bayesian_params = {
    'learning_rate': (0.001, 1),
    'max_depth': (3, 10),
    'num_leaves': (24, 40),
    'min_child_samples': (10, 200),
    'min_child_weight': (1, 50),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1),
    'max_bin': (9, 500),
    'reg_lambda': (0.001, 10),
    'reg_alpha': (0.01, 50),
}

def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators': 100,
        'learning_rate': learning_rate,
        'max_depth': int(round(max_depth)),
        'num_leaves': int(round(num_leaves)),
        'min_child_samples': int(round(min_child_samples)),
        'min_child_weight': int(round(min_child_weight)),
        'subsample': max(min(subsample, 1), 0),
        'colsample_bytree': colsample_bytree,
        'max_bin': max(int(round(max_bin)), 10),
        'reg_lambda': max(reg_lambda, 0),
        'reg_alpha': reg_alpha,
        'objective': 'reg:squarederror',  
    }
    model = XGBRegressor(**params, n_jops=-1)
    model.fit(x_train, y_train,)
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    return results

bay = BayesianOptimization(
    f = xgb_hamsu,
    pbounds=bayesian_params,
    random_state=456,
)

n_iter = 200
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(bay.max)
print(n_iter, '번 걸린시간 : ', round(end_time-start_time, 2), '초')

##########################
# True r2_score : 0.5510983490814336 -> r2_score : 0.564891705174414
# False r2_score : 0.5628181668953597

# 보팅 regressor r2_score : 0.4821744802657928

# 스테킹 cv5 r2_score : 0.3754103844295519
# 스테킹 cv3 r2_score : 0.4367989781168471
'''
{'target': 0.503251985480536, 
'params': {'colsample_bytree': 0.7295276033686791, 'learning_rate': 0.057040912711011776, 'max_bin': 151.60500297216925, 
'max_depth': 6.126388321172584, 'min_child_samples': 19.81943456125996, 'min_child_weight': 37.0858337717953, 'num_leaves': 28.05942418339167, 
'reg_alpha': 31.799267616621826, 'reg_lambda': 7.68144623327445, 'subsample': 0.7214425933867679}}
200 번 걸린시간 :  37.79 초
'''
















