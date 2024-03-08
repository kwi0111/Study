import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import StackingClassifier
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = "C:\\_data\\dacon\\dechul\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0 )
test_csv = pd.read_csv(path + "test.csv", index_col=0 )
submission_csv = pd.read_csv(path + "sample_submission.csv")

# 라벨 엔코더
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder() # 대출기간, 대출목적, 근로기간, 주택소유상태 // 라벨 인코더 : 카테고리형 피처를 숫자형으로 변환
train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = le.fit_transform(train_csv['대출목적'])
train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])

test_csv['주택소유상태'] = le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = le.fit_transform(test_csv['대출목적'])
test_csv['대출기간'] = test_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
test_csv['근로기간'] = le.fit_transform(test_csv['근로기간'])

train_csv['대출등급'] = le.fit_transform(train_csv['대출등급']) # 마지막에 와야함

# x와 y를 분리
x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

x_train, x_test, y_train, y_test = train_test_split(
    x, y , random_state=777, train_size=0.8, stratify=y, shuffle=True
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

# 보팅 소프트 acc_score :  0.842151721273171
# 보팅 하드 acc_score :  0.7953164754140921

# 스테킹 cv = 3 acc_score :  0.8528480191079495
# 스테킹 cv = 5 acc_score :  0.8538864946258892
'''
'''
