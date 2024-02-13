import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

#1. 데이터
# x, y = load_iris(return_X_y=True)
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, train_size=0.8, stratify=y)
print(x_train.shape)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"], "degree":[3,4,5]},   # 4*3 = 12
    {"C":[1,10,100], "kernel":["rbf"], "gamma":[0.001,0.0001]},     # 3*2 = 6
    {"C":[1,10,100,1000], "kernel":["sigmoid"],                     # 4*3*2 = 24
    "gamma":[0.01, 0.001, 0.0001], "degree":[3,4]}
    
]

#2. 모델
print("================ 하빙그리드 ===================")    
# 먼저 전체 데이터의 일부를 사용하여 하이퍼파라미터 조합을 평가하고, 
# 가장 우수한 조합만 선택하여 후보를 줄입니다. 이후 남은 후보들에 대해서도 동일한 과정을 반복하며, 최적의 조합을 찾아냅니다
model = HalvingGridSearchCV(SVC(), 
                     parameters,
                     cv=kfold,
                     verbose=1,
                     refit=True,
                    n_jobs=-1,  # CPU
                    random_state=66,
                    # n_iter=20,  # 디폴트 10
                    factor=3.1, # 디폴트 3 // 후보 모델의 수를 줄이는 비율
                    # min_resources=150,  # 데이터 조절하고싶으면 factor, min_resources 맞춘다.
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



# 최적의 매개변수 : SVC(C=100, degree=5, kernel='linear')
# 최적의 파라미터 : {'C': 100, 'degree': 5, 'kernel': 'linear'}
# best_score : 0.9777777777777779
# model.score : 0.9
# accuracy_score :  0.9
# 최적 튠 ACC :  0.9
# 걸린 시간 : 1.33 초


# iter: 0
# n_candidates: 42
# n_resources: 30
# Fitting 5 folds for each of 42 candidates, totalling 210 fits

# iter: 1
# n_candidates: 14 / 42의 3분할 // 
# n_resources: 90
# Fitting 5 folds for each of 14 candidates, totalling 70 fits

# n_iterations: 2
# n_required_iterations: 4
# n_possible_iterations: 2
# min_resources_: 30 최소 훈련  / 120개 트레인중에서 30개만 쓰겠다.
# max_resources_: 120 최대 훈련 / 트레인 사이즈
# aggressive_elimination: False

# ================ 하빙그리드 ===================
# n_iterations: 2            # 반복 횟수
# n_required_iterations: 3   # 필요한 반복 횟수
# n_possible_iterations: 2   # 가능한 반복 횟수
# min_resources_: 100        # CV * 2 * 라벨의 갯수(10) * 알파
# max_resources_: 1437
# aggressive_elimination: False
# factor: 4


