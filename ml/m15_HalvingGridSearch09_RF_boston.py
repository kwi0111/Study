from sklearn.datasets import load_boston

# 현재 사이킷런 버전 1.3.0 보스턴 안됨. 삭제 후 재설치
#pip uninstall scikit-learn
#pip uninstall scikit-image                  
#pip uninstall scikit-learn-intelex
#pip install scikit-learn==1.1.3


datasets = load_boston()
print(datasets)
x = datasets.data
y = datasets.target
print(x)
print(x.shape) # (506, 13)
print(y.shape) # (506,)

print(datasets.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
# 'B' 'LSTAT']

print(datasets.DESCR) #Describe #행 = Tnstances

# [실습]
#train_size 0.7이상, 0.9이하
#R2 0.8 이상

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore') #warning 무시. 나타내지않음.
import time
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV

#1. 데이터

x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
n_splits=5
#kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
#n_split = 섞어서 분할하는 갯수
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)


#2. 모델구성

parameters = [
    {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1, 2, 4], "min_samples_split": [2, 3, 5, 10]},
]    
     
model = HalvingGridSearchCV(RandomForestRegressor(), 
                     parameters,
                     cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1, 
                     random_state=66,
                     # n_iter=20,  # 디폴트 10
                     factor=5, # 디폴트 3 
                     min_resources=14,  # 데이터 조절하고싶으면 factor, min_resources 맞춘다.
                     )
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

from sklearn.metrics import accuracy_score
best_predict = model.best_estimator_.predict(x_test)
# best_acc_score = accuracy_score(y_test, best_predict)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print('best_score :', model.best_score_)
print('score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
# print("accuracy_score :", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
# print("최적튠 ACC :", accuracy_score(y_test, y_predict))

print("걸린시간 :", round(end_time - start_time, 2), "초")


# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_leaf=3)
# 최적의 파라미터 :  {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 100}
# best_score : 0.993103448275862
# score : 0.9444444444444444
# accuracy_score : 0.9444444444444444
# 최적튠 ACC : 0.9444444444444444

# 최적의 매개변수 :  RandomForestRegressor(min_samples_split=5, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_split': 5}
# best_score : 0.855261505307469
# score : 0.7316671561621604
# 걸린시간 : 2.09 초

# n_iterations: 3
# n_required_iterations: 3
# n_possible_iterations: 3
# min_resources_: 14
# max_resources_: 404
# aggressive_elimination: False
# factor: 5
# ----------
# iter: 0
# n_candidates: 60
# n_resources: 14
# Fitting 5 folds for each of 60 candidates, totalling 300 fits
# ----------
# iter: 1
# n_candidates: 12
# n_resources: 70
# Fitting 5 folds for each of 12 candidates, totalling 60 fits
# ----------
# iter: 2
# n_candidates: 3
# n_resources: 350
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestRegressor(min_samples_split=5, n_jobs=2)
# 최적의 파라미터 :  {'min_samples_split': 5, 'n_jobs': 2}
# best_score : 0.8269972033349603
# score : 0.7049718379041394
# 걸린시간 : 3.24 초