import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time
from sklearn.svm import LinearSVR
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV


#1. 데이터
datasets = fetch_california_housing()
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
    {"RF__n_estimators": [100, 200], "RF__max_depth": [6, 10, 12], "RF__min_samples_leaf": [3, 10]}, # 12
    {"RF__max_depth": [6, 8, 10, 12], "RF__min_samples_leaf": [3, 5, 7, 10]}, # 16
    {"RF__min_samples_leaf": [3, 5, 7, 10], "RF__min_samples_split": [2, 3, 5, 10]}, # 16
    {"RF__min_samples_split": [2, 3, 5, 10]},
    {"RF__min_samples_split": [2, 3, 5, 10]}, # 4
]    
from sklearn.pipeline import Pipeline  # 클래스 /

pipe = Pipeline([('MinMax', MinMaxScaler()),
                  ('RF', RandomForestRegressor())])    # 하나의 파이프가 모델
model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
# model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
# model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1,)

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

from sklearn.metrics import accuracy_score
best_predict = model.best_estimator_.predict(x_test)
#best_acc_score = accuracy_score(y_test, best_predict)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print('best_score :', model.best_score_)
print('score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
#print("accuracy_score :", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
#print("최적튠 ACC :", accuracy_score(y_test, y_predict))

print("걸린시간 :", round(end_time - start_time, 2), "초")

# 최적의 매개변수 :  RandomForestRegressor()
# 최적의 파라미터 :  {'min_samples_split': 2}
# best_score : 0.8047142764400448
# score : 0.6369370382370403
# 걸린시간 : 66.89 초

# 최적의 매개변수 :  RandomForestRegressor(min_samples_split=5, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_split': 5}
# best_score : 0.8027837027790184
# score : 0.6346378595017238
# 걸린시간 : 24.07 초

# n_iterations: 3
# n_required_iterations: 3
# n_possible_iterations: 4
# min_resources_: 100
# max_resources_: 16512
# aggressive_elimination: False
# factor: 5
# ----------
# iter: 0
# n_candidates: 60
# n_resources: 100
# Fitting 5 folds for each of 60 candidates, totalling 300 fits
# ----------
# iter: 1
# n_candidates: 12
# n_resources: 500
# Fitting 5 folds for each of 12 candidates, totalling 60 fits
# ----------
# iter: 2
# n_candidates: 3
# n_resources: 2500
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestRegressor(min_samples_split=3, n_jobs=4)
# 최적의 파라미터 :  {'min_samples_split': 3, 'n_jobs': 4}
# best_score : 0.7550159250321258
# score : 0.6398535812517089
# 걸린시간 : 5.66 초