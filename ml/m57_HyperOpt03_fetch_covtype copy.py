# 랜덤포레스트의 알고리즘이 배깅이다.
# 배깅 - 보팅
# votiong - 모델 여러개 / 같은 데이터 / 소프트 or 하드 / 하드는 다수결 / 소프트는 평균에서 높은쪽
# bagging - 모델 1개 - 데이터가 다르다 (샘플링해서) // 에포 느낌 = n이스터메이트 //  중복 된다 // 
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import time

datasets = fetch_covtype()


x = datasets.data
y= datasets.target
y=y-1

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
# 배깅 True acc_score : 0.7186905673691729 -> n_estimators=100으로 acc_score : 0.718707778628779
# 배깅 False acc_score : 0.7186991729989759

# 보팅 하드 acc_score : 0.8842800960388286
# 보팅 소프트 acc_score : 0.9016634682409232

# 스테킹 cv = 3 acc_score : 0.9625999328760876
# 스테킹 cv = 5 acc_score : 0.9622126795349518
'''
{'target': 0.9464127432166123,
'params': {'colsample_bytree': 0.7992975155268678, 'learning_rate': 0.7421237263532199, 'max_bin': 182.7139061013879, 
'max_depth': 9.719755566307533, 'min_child_samples': 89.46707828279867, 'min_child_weight': 32.080449453411376, 'num_leaves': 25.22788048495924, 
'reg_alpha': 10.232765243501175, 'reg_lambda': 5.411478625710794, 'subsample': 0.6703016341259069}}
70 번 걸린시간 :  688.6 초
'''

'''
best :  {'colsample_bytree': 0.58, 
'learing_rate': 0.633531171000774,
'max_bin': 427.0,
'max_depth': 9.0, 
'min_child_samples': 108.0,
'min_child_weight': 12.0, 
'num_leaves': 36.0, 
'reg_alpha': 2.3692927517790654, 
'reg_lambda': 2.9988343312415995, 
'subsample': 0.8}
Best Accuracy :  0.9504659948538334
100 번 걸린시간 :  487.73 초
'''















