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
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델구성
from xgboost import XGBRegressor
model = XGBRegressor()

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

from sklearn.metrics import r2_score
print('score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score :", r2_score(y_test, y_predict))
print("최적튠 ACC :", r2_score(y_test, y_predict))
print("걸린시간 :", round(end_time - start_time, 2), "초")
print (type(model).__name__, model.feature_importances_)

