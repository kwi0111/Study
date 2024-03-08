import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression # 앤 분류다
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization
import time

#1. 데이터
path = "C:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv)

print(train_csv.shape) #(5497, 13)
print(test_csv.shape) #(1000, 12)
print(submission_csv.shape) #(1000, 2)


print(train_csv.columns) #'quality', 'fixed acidity', 'volatile acidity', 'citric acid',
    #    'residual sugar', 'chlorides', 'free sulfur dioxide',
    #    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
    #    'type'],
    
x = train_csv.drop(['quality'], axis= 1)
y = train_csv['quality']
y = y - 3

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0

test_csv.loc[test_csv['type'] == 'red', 'type'] = 1
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

x_train, x_test, y_train, y_test = train_test_split(
    x, y , random_state=777, train_size=0.8, stratify=y
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
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
    }
    model = XGBClassifier(**params, n_jops=-1)
    model.fit(x_train, y_train,
              eval_set = [(x_train,y_train), (x_test, y_test)],
              eval_metric = 'mlogloss',
              verbose = 0,
              early_stopping_rounds=50,
              )
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
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

##############################################
# 배깅 Ture acc_score :  0.5536363636363636
# 배깅 False acc_score :  0.5518181818181818

# 보팅 하드 acc_score :  0.6772727272727272
# 보팅 소프트 acc_score :  0.6663636363636364

# 스테킹 acc_score :  0.6781818181818182
'''
Trech=0.055, n=12, ACC: 58.09%
Trech=0.064, n=11, ACC: 59.27%
Trech=0.067, n=10, ACC: 60.09%
Trech=0.069, n=9, ACC: 59.27%
Trech=0.070, n=8, ACC: 58.82%
Trech=0.074, n=7, ACC: 58.36%
Trech=0.074, n=6, ACC: 58.00%
Trech=0.077, n=5, ACC: 57.27%
Trech=0.092, n=4, ACC: 57.09%
Trech=0.097, n=3, ACC: 56.18%
Trech=0.114, n=2, ACC: 53.82%
Trech=0.149, n=1, ACC: 54.27%
'''
# {'target': 0.6790909090909091, 'params': {'colsample_bytree': 0.5733936663294885, 'learning_rate': 0.09064338362815302, 
#                                           'max_bin': 58.155571986774056, 'max_depth': 7.635367079986558, 'min_child_samples': 23.75557260578362, 
#                                           'min_child_weight': 1.4128314024820328, 'num_leaves': 33.78126556255404, 'reg_alpha': 0.704026308242926, 
#                                           'reg_lambda': 1.3921166793894182, 'subsample': 0.7444165603721037}}
# 200 번 걸린시간 :  70.24 초