import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

import time
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC #softvector machine
from sklearn.linear_model import Perceptron, LogisticRegression , LinearRegression#분류!
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import time
import tensorflow as tf
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import make_pipeline  # 파이프라인 = 일괄 처리


#1. 데이터

start_time = time.time()

datasets = fetch_covtype()

x = datasets.data
y = datasets.target
#print(x.shape, y.shape) #(581012, 54) (581012,)
#print(pd.value_counts(y))
#print(np.unique(y, return_counts= True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],)
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
#n_split = 섞어서 분할하는 갯수

#2. 모델구성
parameters = [
    {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1, 2, 4], "min_samples_split": [2, 3, 5, 10]},
]    
model = make_pipeline(MinMaxScaler(), GridSearchCV(RandomForestClassifier(), parameters, cv=kfold , n_jobs=-1, refit=True, verbose=1))

#3. 컴파일 및 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

#4. 평가 및 예측
from sklearn.metrics import accuracy_score
results = model.score(x_test, y_test)
print("model.score : ", results)  # acc
y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print("accuracy_score : " , round(acc, 2))

print("걸린시간 :", round(end_time - start_time, 2), "초")

# n_iterations: 3
# n_required_iterations: 3
# n_possible_iterations: 5
# min_resources_: 300
# max_resources_: 464809
# aggressive_elimination: False
# factor: 5
# ----------
# iter: 0
# n_candidates: 60
# n_resources: 300
# Fitting 5 folds for each of 60 candidates, totalling 300 fits
# ----------
# iter: 1
# n_candidates: 12
# n_resources: 1500
# Fitting 5 folds for each of 12 candidates, totalling 60 fits
# ----------
# iter: 2
# n_candidates: 3
# n_resources: 7500
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestClassifier()
# 최적의 파라미터 :  {'min_samples_split': 2}
# best_score : 0.7873033577940849
# score : 0.949304234830426
# accuracy_score : 0.949304234830426
# 최적튠 ACC : 0.949304234830426
# 걸린시간 : 62.49 초

