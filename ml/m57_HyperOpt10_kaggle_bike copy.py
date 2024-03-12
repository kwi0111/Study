
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
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
        'n_estimators' : 500,
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

'''
best :  {'colsample_bytree': 0.92,
'learing_rate': 0.9954192966632852,
'max_bin': 198.0, 'max_depth': 10.0,
'min_child_samples': 200.0, 
'min_child_weight': 39.0, 
'num_leaves': 24.0, 
'reg_alpha': 8.981694699050436,
'reg_lambda': 0.11244728343954247,
'subsample': 0.52}
Best R2_score :  0.3326001598556524
100 번 걸린시간 :  3.3 초
'''














