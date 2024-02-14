import numpy as np
import pandas as pd #판다스에 데이터는 넘파이 형태로 들어가있음.
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import time
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV

#1. 데이터
path = "C:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")

print(train_csv.columns)
# ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#       'humidity', 'windspeed', 'casual', 'registered', 'count']

print(train_csv.info())
print(test_csv.info())

x = train_csv.drop(['casual','registered','count'], axis=1)
y = train_csv['count']

from sklearn.model_selection import train_test_split,KFold,cross_val_score, GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#2. 모델구성
from xgboost import XGBRegressor
model = XGBRegressor(cv=5, verbose=1,)

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

from sklearn.metrics import r2_score
print('r2_score :', model.score(x_test, y_test))
y_predict = model.predict(x_test)
print("걸린시간 :", round(end_time - start_time, 2), "초")

