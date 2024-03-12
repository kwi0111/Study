
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
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
search_space = {
    'learning_rate' : hp.uniform('learing_rate', 0.001, 1),
    'max_depth' : hp.quniform('max_depth', 3, 10, 1.0),
    'num_leaves' : hp.quniform('num_leaves', 24, 40, 1),
    'min_child_samples' : hp.quniform('min_child_samples', 10, 200, 1),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 50, 1),
    'subsample' : hp.quniform('subsample', 0.5, 1, 0.01),
    'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.01),
    'max_bin' : hp.quniform('max_bin', 9, 500, 1),
    'reg_lambda' : hp.uniform('reg_lambda', -0.001, 10),
    'reg_alpha' : hp.uniform('reg_alpha', 0.01, 50),
}

def xgb_hamsu(search_space):
    params = {
        'n_estimators' : 100,
        'learning_rate' : search_space['learning_rate'],
        'max_depth' : int(search_space['max_depth']),        # 무조건 정수형
        'num_leaves' : int(search_space['num_leaves']),
        'min_child_samples' : int(search_space['min_child_samples']),
        'min_child_weight' : int(search_space['min_child_weight']),
        'subsample' : max(min(search_space['subsample'], 1), 0),    # 0~1 사이의 값
        'colsample_bytree' : search_space['colsample_bytree'],
        'max_bin' : max(int(search_space['max_bin']), 10),   # 무조건 10이상
        'reg_lambda' : max(search_space['reg_lambda'], 0),          # 무조건 양수만
        'reg_alpha' : search_space['reg_alpha'],
    }
    model = XGBRegressor(**params, n_jobs=-1)
    model.fit(x_train, y_train)
  
    
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    return results

trial_val = Trials()

n_iter = 100
start_time = time.time()
best = fmin(
    fn = xgb_hamsu,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trial_val,
    rstate=np.random.default_rng(seed=10)
)
end_time = time.time()

best_r2_score = max(trial_val.losses())  # 최적 R2 값을 가져옵니다

print('best : ', best)
print("Best R2_score : ", best_r2_score)
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
'''
best :  {'colsample_bytree': 0.89,
'learing_rate': 0.006548881828335525,
'max_bin': 249.0,
'max_depth': 10.0,
'min_child_samples': 199.0,
'min_child_weight': 41.0,
'num_leaves': 40.0,
'reg_alpha': 11.729205625458356,
'reg_lambda': 3.1836071961166383,
'subsample': 0.54}
Best R2_score :  0.8373652108663727
100 번 걸린시간 :  4.13 초
'''















