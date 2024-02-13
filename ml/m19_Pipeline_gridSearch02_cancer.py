import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV
import time
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline  # 파이프라인 = 일괄 처리 / 함수
from sklearn.pipeline import Pipeline  # 클래스 /


#1.데이터
datasets = load_breast_cancer()

x = datasets.data
y= datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델구성
parameters = [
    {"RF__n_estimators": [100, 200], "RF__max_depth": [6, 10, 12], "RF__min_samples_leaf": [3, 10]}, # 12
    {"RF__max_depth": [6, 8, 10, 12], "RF__min_samples_leaf": [3, 5, 7, 10]}, # 16
    {"RF__min_samples_leaf": [3, 5, 7, 10], "RF__min_samples_split": [2, 3, 5, 10]}, # 16
    {"RF__min_samples_split": [2, 3, 5, 10]},
    {"RF__min_samples_split": [2, 3, 5, 10]}, # 4
]    
     
pipe = Pipeline([('MinMax', MinMaxScaler()),
                  ('RF', RandomForestClassifier())])    # 하나의 파이프가 모델
# model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
# model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1,)

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

from sklearn.metrics import accuracy_score
best_predict = model.best_estimator_.predict(x_test)
best_acc_score = accuracy_score(y_test, best_predict)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print('best_score :', model.best_score_)
print('score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적튠 ACC :", accuracy_score(y_test, y_predict))

print("걸린시간 :", round(end_time - start_time, 2), "초")

# n_iterations: 3
# n_required_iterations: 4
# n_possible_iterations: 3
# min_resources_: 30
# max_resources_: 455
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 60
# n_resources: 30
# Fitting 5 folds for each of 60 candidates, totalling 300 fits
# ----------
# iter: 1
# n_candidates: 20
# n_resources: 90
# Fitting 5 folds for each of 20 candidates, totalling 100 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 270
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# 최적의 매개변수 :  RandomForestClassifier(max_depth=10, min_samples_leaf=3)
# 최적의 파라미터 :  {'max_depth': 10, 'min_samples_leaf': 3, 'n_estimators': 100}
# best_score : 0.962962962962963
# score : 0.9473684210526315
# accuracy_score : 0.9473684210526315
# 최적튠 ACC : 0.9473684210526315
# 걸린시간 : 3.86 초

# 최적의 매개변수 :  Pipeline(steps=[('MinMax', MinMaxScaler()), ('RF', RandomForestClassifier())])
# 최적의 파라미터 :  {'RF__min_samples_split': 2}
# best_score : 0.9692307692307693
# score : 0.9473684210526315
# accuracy_score : 0.9473684210526315
# 최적튠 ACC : 0.9473684210526315
# 걸린시간 : 19.85 초

# 최적의 매개변수 :  Pipeline(steps=[('MinMax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestClassifier(min_samples_leaf=3,
#                                         min_samples_split=5))])
# 최적의 파라미터 :  {'RF__min_samples_split': 5, 'RF__min_samples_leaf': 3}
# best_score : 0.964835164835165
# score : 0.956140350877193
# accuracy_score : 0.956140350877193
# 최적튠 ACC : 0.956140350877193
# 걸린시간 : 4.22 초

# 최적의 매개변수 :  Pipeline(steps=[('MinMax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestClassifier(max_depth=8, min_samples_leaf=5))])
# 최적의 파라미터 :  {'RF__max_depth': 8, 'RF__min_samples_leaf': 5}
# best_score : 0.9277777777777778
# score : 0.9473684210526315
# accuracy_score : 0.9473684210526315
# 최적튠 ACC : 0.9473684210526315
# 걸린시간 : 21.12 초