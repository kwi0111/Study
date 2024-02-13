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
    {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1, 2, 4], "min_samples_split": [2, 3, 5, 10]},
]    
     
model = HalvingGridSearchCV(RandomForestClassifier(), 
                     parameters,
                     cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1, 
                     random_state=66,
                     # n_iter=20,  # 디폴트 10
                     factor=3, # 디폴트 3 
                     min_resources=30,  # 데이터 조절하고싶으면 factor, min_resources 맞춘다.
                     )
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