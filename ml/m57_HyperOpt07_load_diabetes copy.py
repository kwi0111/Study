
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import time

datasets = load_diabetes()


x = datasets['data']
y = datasets.target


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

'''
best :  {'colsample_bytree': 0.77, 
'learing_rate': 0.995965869973553,
'max_bin': 157.0, 
'max_depth': 4.0,
'min_child_samples': 123.0, 
'min_child_weight': 20.0, 
'num_leaves': 39.0,
'reg_alpha': 4.255419398696731,
'reg_lambda': 4.338171818535722, 
'subsample': 0.6900000000000001}
Best R2_score :  0.4763424493020122
100 번 걸린시간 :  1.59 초
'''















