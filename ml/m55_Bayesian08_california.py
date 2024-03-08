
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
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

#1. 데이터
datasets = fetch_california_housing()
x = np.array(datasets.data)
y = np.array(datasets.target)


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

# 리니어 Ture r2_score : 0.5827784121245206
# 랜포 True 최종점수 0.8021630262259856

# 보팅 regressor r2_score : 0.7559852381903986

# 스테킹 cv3 r2_score : 0.8465688911293758
# 스케팅 cv5 r2_score : 0.8469958475025819
'''
{'target': 0.8444267636171674, 
'params': {'colsample_bytree': 0.9553298281531644, 'learning_rate': 0.31750197522060913, 'max_bin': 211.37385834155933, 
'max_depth': 9.737787106280948, 'min_child_samples': 171.97470828133925, 'min_child_weight': 21.73167390419969, 'num_leaves': 36.40795293895731, 
'reg_alpha': 6.863865561556281, 'reg_lambda': 3.424059930585913, 'subsample': 0.7246044633012999}}
200 번 걸린시간 :  42.81 초
'''
















