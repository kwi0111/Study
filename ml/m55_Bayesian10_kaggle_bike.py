
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
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
path = "C:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")
print(train_csv.columns)
# ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#       'humidity', 'windspeed', 'casual', 'registered', 'count']
x = train_csv.drop(['casual','registered','count'], axis=1)
y = train_csv['count']

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
# 100, True acc_score : 0.31887104549317047
# 100, False acc_score : 0.2642653866306752

# 보팅 regressor acc_score : 0.3288664868959942

# 스테킹 cv5 acc_score : 0.33458892928987394
# 스테킹 cv3  acc_score : 0.3405065952773445
'''
{'target': 0.3399617310031211,
'params': {'colsample_bytree': 0.9252258899499366, 'learning_rate': 0.17396005100635592, 'max_bin': 26.046105485288695, 
'max_depth': 5.2506874442998654, 'min_child_samples': 81.71687277704271, 'min_child_weight': 36.36382943777442, 'num_leaves': 27.87886184736804,
'reg_alpha': 35.011045806752975, 'reg_lambda': 3.4144383711420394, 'subsample': 0.9202896488261956}}        
200 번 걸린시간 :  38.35 초
'''
















