import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import sklearn as sk
print(sk.__version__) # 1.3 버젼은 안돌아간다.
print(sk.__version__) # 1.1.3 버젼은 돌아간다.

#1. 데이터
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
print("================ HalvingRandomSearchCV ===================")    
# 먼저 전체 데이터의 일부를 사용하여 하이퍼파라미터 조합을 평가하고, 
# 가장 우수한 조합만 선택하여 후보를 줄입니다. 이후 남은 후보들에 대해서도 동일한 과정을 반복하며, 최적의 조합을 찾아냅니다
# model = HalvingGridSearchCV(SVC(), 
model = HalvingRandomSearchCV(SVC(), 
                     parameters,
                     cv=kfold,
                     verbose=1,
                     refit=True,
                    n_jobs=-1,  # CPU
                    random_state=66,
                    # n_iter=20,  # 디폴트 10
                    factor=3.1, # 디폴트 3 // 후보 모델의 수를 줄이는 비율
                    min_resources=150,  # 데이터 조절하고싶으면 factor, min_resources 맞춘다.
                     )
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 :", model.best_estimator_)
print("최적의 파라미터 :", model.best_params_)
print("best_score :", model.best_score_)
print("model.score :", model.score(x_test, y_test))

y_predict =model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 ACC : ", accuracy_score(y_test, y_pred_best))
print("걸린 시간 :", round(end_time - start_time, 2),"초")




