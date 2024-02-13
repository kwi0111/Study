import numpy as np
from sklearn.datasets import load_iris, load_digits, load_diabetes
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

#1. 데이터
# x, y = load_iris(return_X_y=True)
# x, y = load_digits(return_X_y=True)
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, train_size=0.8)
# print(x_train.shape)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"], "degree":[3,4,5]},   # 12
    {"C":[1,10,100], "kernel":["rbf"], "gamma":[0.001,0.0001]},     # 6
    {"C":[1,10,100,1000], "kernel":["sigmoid"],                     # 24
    "gamma":[0.01, 0.001, 0.0001], "degree":[3,4]}
    
]

#2. 모델
print("================ 하빙그리드 ===================")
model = HalvingGridSearchCV(SVR(), 
                     parameters,
                     cv=kfold,
                     verbose=1,
                     refit=True,
                    n_jobs=-1,
                    random_state=66,
                    # n_iter=20,  # 디폴트 10
                    factor=3, # 디폴트 3 
                    # min_resources=150,  # 데이터 조절하고싶으면 factor, min_resources 맞춘다.
                     )
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 :", model.best_estimator_)
# 최적의 매개변수 : SVC(C=1, kernel='linear') 
print("최적의 파라미터 :", model.best_params_)
# 최적의 파라미터 : {'C': 1, 'degree': 3, 'kernel': 'linear'} 우리가 지정한것 중 베스트
print("best_score :", model.best_score_)
print("model.score :", model.score(x_test, y_test))

y_predict =model.predict(x_test)
# print("accuracy_score : ", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
                # SVC(C=1, kernel='linear').predict(x_test)
                
# print("최적 튠 ACC : ", accuracy_score(y_test, y_pred_best))

print("걸린 시간 :", round(end_time - start_time, 2),"초")

# ================ 하빙그리드 ===================
# n_iterations: 2            # 반복 횟수
# n_required_iterations: 3   # 필요한 반복 횟수
# n_possible_iterations: 2   # 가능한 반복 횟수
# min_resources_: 100        # CV * 2 * 알파
# max_resources_: 1437
# aggressive_elimination: False
# factor: 4



