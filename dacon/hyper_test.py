from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedStratifiedKFold, GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
import tensorflow as tf
import pandas as pd
import numpy as np
import optuna
import random
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import datetime
import time
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42  
tf.random.set_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

path = 'C:\\_data\\dacon\\hyper\\'
train_csv = pd.read_csv(path + 'train.csv')
submission_csv = pd.read_csv(path + "sample_submission.csv") 
print(train_csv)

# person_id: 유저별 고유 아이디
# Sex: 성별
# past_login_total: 과거(5월 8일 이전)에 로그인한 총 횟수
# past_1_month_login: 과거 1달간 로그인한 총 횟수
# past_1_week_login: 과거 1주간 로그인한 총 횟수
# sub_size: 과거에 데이콘 대회에서의 총 제출 수
# email_type: 가입한 이메일 종류
# phone_rat: 폰으로 접속한 비율
# apple_rat: 애플 기기로 접속한 비율
# login: 로그인 여부 

# person_id 컬럼 제거
x = train_csv.drop('login', axis=1)
y = train_csv['login']

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= RANDOM_STATE, train_size=0.8, stratify=y)

# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# x_test = scaler.transform(x_test)


n_splits=3
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

from skopt import BayesSearchCV
# Bayesian Optimization을 위한 파라미터 공간 정의

param_grid = {
    'n_estimators': list(range(10, 101, 1)),  # 10부터 1000까지 50 단위로
    'max_depth': [None] + list(range(1, 16, 5)),  # None 포함, 6부터 25까지 5 단위로
    'min_samples_split': [2, 5],  # 분할을 위한 최소 샘플 수
    'min_samples_leaf': [1, 2, 4],  # 리프 노드가 가져야 하는 최소 샘플 수
    'max_features': ['auto', 'sqrt', 'log2'],  # 최대 피처 개수
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2],  # 리프 노드에 있어야 하는 가중치의 최소 합
    'max_leaf_nodes': [None] + list(range(5, 21, 5)),  # 최대 리프 노드 수
    'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1],   # 노드를 분할하기 위한 불순도 감소량 최소값
    # 'criterion': ['gini'],
    # "bootstrap": [True],
}

# RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state=42)

# GridSearchCV 객체 생성
model = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kfold, n_jobs=-1, verbose=1, scoring='roc_auc', refit=True)

# # BayesSearchCV 객체 생성
# model = BayesSearchCV(
#     estimator=rf,
#     search_spaces=param_space,
#     n_iter=100,
#     cv=kfold,
#     scoring='roc_auc',
#     n_jobs=-1,
#     verbose=1,
#     refit=True
# )


# GridSearchCV를 사용한 학습
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

# 최적의 파라미터와 최고 점수 출력
best_params = model.best_params_
best_score = model.best_score_

best_predict = model.best_estimator_.predict(x_test)
accuracy = accuracy_score(y_test, best_predict)


# 훈련 데이터로 AUC 계산
y_pred_proba = model.predict_proba(x_test)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_proba)

y_predict = model.predict(x_test)

y_pred_best = model.best_estimator_.predict(x_test)
print("최적의 매개변수: ", model.best_estimator_)
print("최적의 파라미터: ", best_params)
print("최고 교차 검증 점수: ", best_score)
print("테스트 데이터 정확도: ", accuracy)
print("테스트 데이터 AUC: ", test_auc)
print("걸린 시간: ", round(end_time - start_time, 2), "초")

submission_csv = pd.read_csv('C:\\_data\\dacon\\hyper\\sample_submission.csv')
for label in submission_csv:
    if label in best_params.keys():
        submission_csv[label] = best_params[label]
    
import datetime
dt = datetime.datetime.now()
submission_csv.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_AUC_{test_auc:4}.csv",index=False)


