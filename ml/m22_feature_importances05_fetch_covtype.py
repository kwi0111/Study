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

#1. 데이터

start_time = time.time()

datasets = fetch_covtype()

x = datasets.data
y = datasets.target
y = y - 1
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

#2. 모델구성
from xgboost import XGBClassifier
model = XGBClassifier(cv=5, verbose=1,)
# model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
# model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1,)

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()
 
from sklearn.metrics import accuracy_score
print('score :', model.score(x_test, y_test))
y_predict = model.predict(x_test)

print("accuracy_score :", accuracy_score(y_test, y_predict))
print("최적튠 ACC :", accuracy_score(y_test, y_predict))
print("걸린시간 :", round(end_time - start_time, 2), "초")
print(type(model).__name__, model.feature_importances_)


