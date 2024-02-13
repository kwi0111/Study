# factor 조절
# min_resources 조절
# 훈련 데이터를 iter_2이상 돌려라

import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, train_size=0.8, stratify=y)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = {
    "n_estimators": [10, 50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

#2. 모델
model = HalvingGridSearchCV(RandomForestClassifier(), 
                     parameters,
                     cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1, 
                     random_state=66,
                     # n_iter=20,  # 디폴트 10
                     factor=3, # 디폴트 3 
                     min_resources=5,  # 데이터 조절하고싶으면 factor, min_resources 맞춘다.
                     )
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 :", model.best_estimator_)
# 최적의 매개변수 : SVC(C=1, kernel='linear') 
print("최적의 파라미터 :", model.best_params_)
# 최적의 파라미터 : {'C': 1, 'degree': 3, 'kernel': 'linear'} 우리가 지정한것 중 베스트

print("best_score :", model.best_score_)    # train 스코어
# best_score : 0.975
print("model.score :", model.score(x_test, y_test))
# model.score : 0.9666666666666667

y_predict =model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
                # SVC(C=1, kernel='linear').predict(x_test)
                
print("최적 튠 ACC : ", accuracy_score(y_test, y_pred_best))

print("걸린 시간 :", round(end_time - start_time, 2),"초")

# n_iterations: 3
# n_required_iterations: 5
# n_possible_iterations: 3
# min_resources_: 5
# max_resources_: 120
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 144
# n_resources: 5
# Fitting 5 folds for each of 144 candidates, totalling 720 fits
# ----------
# iter: 1
# n_candidates: 48
# n_resources: 15
# Fitting 5 folds for each of 48 candidates, totalling 240 fits
# ----------
# iter: 2
# n_candidates: 16
# n_resources: 45









