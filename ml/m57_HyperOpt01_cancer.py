
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
from bayes_opt import BayesianOptimization
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y= datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size= 0.8,  stratify=y)

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
              eval_metric = 'logloss',
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

# {'target': 0.9912280701754386, 'params': {'colsample_bytree': 0.6485270689906658, 'learning_rate': 0.7641361232321419, 
#                                           'max_bin': 461.3155414748959, 'max_depth': 9.581742007477946, 'min_child_samples': 23.232180287251843, 
#                                           'min_child_weight': 1.545202228210833, 'num_leaves': 25.23719419567511, 
#                                           'reg_alpha': 4.802187917493291, 'reg_lambda': 5.880056873489918, 'subsample': 0.6177289425160959}}
# 100 번 걸린시간 :  17.46 초
# 원하는건 파라미터와 결과치

'''
 50/50 [00:02<00:00, 24.98trial/s, best loss: 0.01754385964912286]
best :  {'colsample_bytree': 0.66, 
'learing_rate': 0.9761323116118479, 
'max_bin': 417.0, 'max_depth': 8.0, 
'min_child_samples': 134.0, 
'min_child_weight': 10.0, 
'num_leaves': 38.0, 
'reg_alpha': 10.092387629631173, 
'reg_lambda': 9.43588306152347, 
'subsample': 0.7000000000000001}

Best Accuracy :  0.9824561403508771
100 번 걸린시간 :  2.0 초
'''














