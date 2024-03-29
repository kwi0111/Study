from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedStratifiedKFold, GroupKFold 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
import tensorflow as tf
import pandas as pd
import numpy as np
import optuna
import random
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel
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
# 데이터 전처리 과정이 끝난 학습 데이터 (추가 데이터 전처리 과정을 진행하지 않습니다.)
# RF 모델 하이퍼파라미터를 제출 시, 해당 데이터로 자동적으로 학습됩니다.
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
x = train_csv.drop(['person_id', 'login'], axis=1)
y = train_csv['login']

# x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 42, train_size=0.75, stratify=y)

# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# x_test = scaler.transform(x_test)


n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
# GridSearchCV를 위한 하이퍼파라미터 설정
# param_grid = {
#     "n_estimators": [150, 200, 250],     # 범위: 10 ~ 1000 사이의 양의 정수. 일반적으로 값이 클수록 모델 성능이 좋아지지만, 계산 비용과 시간도 증가합니다.
#     "criterion": ['gini', 'entropy'],   # 옵션: 'gini', 'entropy'. 'gini'는 진니 불순도를, 'entropy'는 정보 이득을 기준으로 합니다.
#     "max_depth": [6, 8, 10, None],         # 범위: None 또는 양의 정수. None으로 설정하면 노드가 모든 리프가 순수해질 때까지 확장됩니다. 양의 정수를 설정하면 트리의 최대 깊이를 제한합니다.
#     "min_samples_split": [2, 5, 10],    # 범위: 2 이상의 정수 또는 0과 1 사이의 실수 (비율을 나타냄, (0, 1] ). 내부 노드를 분할하기 위해 필요한 최소 샘플 수를 지정합니다.
#     "min_samples_leaf": [1, 3, 8],      # 범위: 1 이상의 정수 또는 0과 0.5 사이의 실수 (비율을 나타냄, (0, 0.5] ). 리프 노드가 가져야 하는 최소 샘플 수를 지정합니다.
#     "min_weight_fraction_leaf": [0, 0.2],   # 범위: 0.0에서 0.5 사이의 실수. 리프 노드에 있어야 하는 샘플의 최소 가중치 비율을 지정합니다.
#     "max_features": ['auto'],       # 옵션: 'auto', 'sqrt', 'log2', None 또는 양의 정수/실수. 최적의 분할을 찾기 위해 고려할 특성의 수 또는 비율을 지정합니다.
#     "max_leaf_nodes": [None, 10],   # 범위: None 또는 양의 정수. 리프 노드의 최대 수를 제한합니다. None은 무제한을 의미합니다.
#     "min_impurity_decrease": [0, 0.1],  # 범위: 0.0 이상의 실수. 노드를 분할할 때 감소해야 하는 불순도의 최소량을 지정합니다.
#     "bootstrap": [True, False]  # 옵션: True, False. True는 부트스트랩 샘플을 사용하여 개별 트리를 학습시킵니다. False는 전체 데이터셋을 사용하여 각 트리를 학습시킵니다
# }
# param_grid = {
#     "n_estimators": [150, 200, 250],     # 최적 값: 200
#     "criterion": ['gini', 'entropy'],   # 최적 값: 'entropy'
#     "max_depth": [6, 8, 10, None],         # 최적 값: None
#     "min_samples_split": [2, 5, 10],    # 최적 값: 2
#     "min_samples_leaf": [1, 3, 8],      # 최적 값: 1
#     "min_weight_fraction_leaf": [0, 0.2],   # 최적 값: 0
#     "max_features": ['auto'],       # 최적 값: 'auto'
#     "max_leaf_nodes": [None, 10],   # 최적 값: None
#     "min_impurity_decrease": [0, 0.1],  # 최적 값: 0
#     "bootstrap": [True, False]  # 최적 값: True
# }
from skopt import BayesSearchCV
# Bayesian Optimization을 위한 파라미터 공간 정의
# param_space = {
#     "n_estimators": (150, 250),          # 최적 값: 200
#     "criterion": ['gini', 'entropy'],    # 최적 값: 'entropy'
#     "max_depth": (6, 10),                 # 최적 값: None
#     "min_samples_split": (2, 10),         # 최적 값: 2
#     "min_samples_leaf": (1, 8),           # 최적 값: 1
#     "min_weight_fraction_leaf": (0, 0.2), # 최적 값: 0
#     "max_features": ['auto'],             # 최적 값: 'auto'
#     "max_leaf_nodes": (2, 10),         # 최적 값: None
#     "min_impurity_decrease": (0, 0.1),    # 최적 값: 0
#     "bootstrap": [True, False]            # 최적 값: True
# }

param_space = {
    "n_estimators": (100, 1500),          # 트리의 개수 범위를 100에서 1500 사이로 변경
    "max_depth": (5, 200),               # 트리의 최대 깊이 범위를 5에서 200 사이로 변경
    "min_samples_split": (2, 100),       # 노드를 분할하기 위한 최소 샘플 수 범위를 2에서 100 사이로 변경
    "min_samples_leaf": (1, 50),         # 리프 노드가 가져야 하는 최소 샘플 수 범위를 1에서 50 사이로 변경
    "max_features": ['auto', 'sqrt', 'log2', None],  # None을 포함한 다양한 옵션 유지
    "max_leaf_nodes": (10, 500),         # 리프 노드의 최대 수 범위를 10에서 500 사이로 변경
    "bootstrap": [True],
    "min_impurity_decrease": (0, 0.2),   # 불순도 감소 범위를 0에서 0.2로 변경
    "min_weight_fraction_leaf": (0, 0.5), # 최소 가중치 비율 범위를 0에서 0.5로 변경
    "criterion": ['gini'],
}

# RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state=42)

# GridSearchCV 객체 생성
# model = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kfold, n_jobs=-1, verbose=0, scoring='roc_auc', refit=True)

# BayesSearchCV 객체 생성
model = BayesSearchCV(
    estimator=rf,
    search_spaces=param_space,
    n_iter=100,
    cv=kfold,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=0,
    refit=True
)


# GridSearchCV를 사용한 학습
start_time = time.time()
model.fit(x, y)
end_time = time.time()

# 최적의 파라미터와 최고 점수 출력
best_params = model.best_params_
best_score = model.best_score_


from sklearn.metrics import accuracy_score
best_predict = model.best_estimator_.predict(x)
best_acc_score = accuracy_score(y, best_predict)


print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print('best_score :', model.best_score_)
print('score :', model.score(x, y))

# 훈련 데이터로 AUC 계산
y_pred_proba = model.predict_proba(x)[:, 1]
train_auc = roc_auc_score(y, y_pred_proba)

y_predict = model.predict(x)

y_pred_best = model.best_estimator_.predict(x)
# print("최적튠 ACC :", accuracy_score(y, y_predict))
print("accuracy_score :", accuracy_score(y, y_predict))
print("걸린시간 :", round(end_time - start_time, 2), "초")


# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submission_csv.columns:
        submission_csv[param] = value

print(submission_csv)
# print(submission_csv.shape) # (1, 10)
dt = datetime.datetime.now()
submission_csv.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_acc_{best_acc_score:4}.csv",index=False)

print('AUC on training data:', train_auc)

