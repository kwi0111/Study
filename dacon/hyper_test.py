import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import time
import datetime
from hyperopt import hp, fmin, tpe, Trials
import numpy as np

# 데이터 불러오기
path = 'C:\\_data\\dacon\\hyper\\'
train_csv = pd.read_csv(path + 'train.csv')
submission_csv = pd.read_csv(path + "sample_submission.csv") 

# person_id 컬럼 제거
x = train_csv.drop(['person_id', 'login'], axis=1)
y = train_csv['login']

# 데이터 스케일링
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Stratified KFold를 사용한 교차 검증 설정
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# 하이퍼파라미터 탐색 함수 수정
def objective(params):
    # RandomForestClassifier 모델 생성 전에 criterion 값을 처리
    if isinstance(params['criterion'], int):
        if params['criterion'] == 0:
            params['criterion'] = 'gini'
        elif params['criterion'] == 1:
            params['criterion'] = 'entropy'
        else:
            raise ValueError("Invalid criterion value")

    # max_features를 문자열로 변환
    params['max_features'] = ['auto', 'sqrt', 'log2', None][params['max_features']]

    # bootstrap을 문자열로 변환
    params['bootstrap'] = bool(params['bootstrap'])

    # n_estimators와 max_depth를 정수로 변환
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])

    # min_samples_split을 정수로 변환
    params['min_samples_split'] = int(params['min_samples_split'])

    # max_leaf_nodes를 정수로 변환
    params['max_leaf_nodes'] = int(params['max_leaf_nodes'])

    # RandomForestClassifier 모델 생성
    model = RandomForestClassifier(**params, random_state=42)

    # 교차 검증을 통한 모델 성능 측정
    auc_scores = []
    for train_idx, val_idx in kfold.split(x, y):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(x_train, y_train)
        y_pred_proba = model.predict_proba(x_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, y_pred_proba))

    # 교차 검증 결과의 평균 AUC 반환 (Hyperopt는 손실을 최소화하는 방향으로 탐색)
    return -np.mean(auc_scores)



# 하이퍼파라미터 탐색 공간 설정 
param_space = {
    "n_estimators": hp.quniform("n_estimators", 100, 1500, 1),  # 정수형으로 설정
    "max_depth": hp.quniform("max_depth", 5, 200, 1),
    "min_samples_split": hp.quniform("min_samples_split", 2, 100, 1),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 50, 1),
    "max_features": hp.choice("max_features", ['auto', 'sqrt', 'log2', None]),
    "max_leaf_nodes": hp.quniform("max_leaf_nodes", 10, 500, 1),
    "min_impurity_decrease": hp.uniform("min_impurity_decrease", 0, 0.2),
    "min_weight_fraction_leaf": hp.uniform("min_weight_fraction_leaf", 0, 0.5),
    "bootstrap": hp.choice("bootstrap", [True]),
    "criterion": hp.choice("criterion", ['gini']),
}


# Trials 객체 생성
trials = Trials()

# fmin 함수를 사용하여 하이퍼파라미터 탐색 실행
best_params = fmin(fn=objective,
                   space=param_space,
                   algo=tpe.suggest,
                   max_evals=10,
                   trials=trials)

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submission_csv.columns:
        # criterion, max_features, bootstrap은 인덱스로 변환된 값에서 실제 값으로 변환
        if param == 'criterion':
            value = 'gini' if value == 0 else 'entropy'
        elif param == 'max_features':
            value = ['auto', 'sqrt', 'log2', None][value]
        elif isinstance(value, float):  # 정수형이 아니라면 정수로 변환
            value = int(value)
        elif param in ['bootstrap', 'criterion', 'max_features']:
            value = 'False' if value == 0 else 'True' if value == 1 else value  # 0이면 "False", 1이면 "True"
        submission_csv[param] = str(value)  # 문자열로 변환하여 저장


# from sklearn.metrics import accuracy_score
# best_predict = model.best_estimator_.predict(x)
# best_acc_score = accuracy_score(y, best_predict)

# 최적의 하이퍼파라미터 출력
print("Best Hyperparameters:", best_params)

submission_csv.fillna(value="None", inplace=True)
# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submission_csv.columns:
        submission_csv[param] = value

print(submission_csv)
# print(submission_csv.shape) # (1, 10)
dt = datetime.datetime.now()
submission_csv.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}.csv",index=False)
