
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')

#1.
datasets = load_digits()


x = datasets.data
y= datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size= 0.8, stratify=y)

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
    model = XGBClassifier(**params, n_jobs=-1)
    model.fit(x_train, y_train,
              eval_set = [(x_train,y_train), (x_test, y_test)],
              eval_metric = 'mlogloss',
              verbose = 0,
              early_stopping_rounds=100,
              )
    
    y_predict = model.predict(x_test)
    # results = accuracy_score(y_test, y_predict)
    results = 1 - accuracy_score(y_test, y_predict)
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

best_accuracy = min(trial_val.losses())
best_trial = trial_val.best_trial

print('best : ', best)
print("Best Accuracy : ", 1 - best_accuracy)  # 최대 정확도를 출력
print(n_iter, '번 걸린시간 : ', round(end_time-start_time, 2), '초')

##########################
# acc_score : 0.9555555555555556

# 보팅 하드 acc_score : 0.975
# 보팅 소프트 acc_score : acc_score : 0.9805555555555555

# 스테킹 acc_score : 0.9777777777777777
'''
{'target': 0.9833333333333333, 'params': {'colsample_bytree': 0.528545942122125, 'learning_rate': 0.2742422199116809, 
'max_bin': 265.0286610060061, 'max_depth': 9.574201698925368, 'min_child_samples': 141.63248019138757, 'min_child_weight': 1.0282884424019545, 
'num_leaves': 33.86761254220593, 'reg_alpha': 0.6281262034934328, 'reg_lambda': 8.620492448624052, 'subsample': 0.839665404628686}}
200번 걸린 시간: 59.42초
'''

'''
best :  {'colsample_bytree': 0.75,
'learing_rate': 0.2505417204017185, 
'max_bin': 141.0, 'max_depth': 6.0, 
'min_child_samples': 130.0, 
'min_child_weight': 3.0, 
'num_leaves': 30.0, 
'reg_alpha': 1.4887080389678211,
'reg_lambda': 6.30841258828627,
'subsample': 0.86}
Best Accuracy :  0.9722222222222222
100 번 걸린시간 :  8.61 초
'''















