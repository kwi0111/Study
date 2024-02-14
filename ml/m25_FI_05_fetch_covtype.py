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
y = y -1
#print(x.shape, y.shape) #(581012, 54) (581012,)
#print(pd.value_counts(y))
#print(np.unique(y, return_counts= True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],)
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

''' 25퍼 미만 열 삭제 '''
columns = datasets.feature_names
# columns = x.columns
x = pd.DataFrame(x,columns=columns)
print("x.shape",x.shape)
# ''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
fi_str = "0.0954867  0.00713124 0.00411837 0.01321335 0.00743847 0.01333909\
 0.00860075 0.01177858 0.00563742 0.0125805  0.05910649 0.02524424\
 0.03462541 0.02239    0.00331126 0.04878221 0.01951258 0.04187524\
 0.00532929 0.00457224 0.00148059 0.01068641 0.00853365 0.01288974\
 0.0110577  0.04379934 0.01060384 0.00372109 0.00081446 0.00784765\
 0.01112258 0.00627568 0.00522331 0.01550328 0.0191748  0.05527256\
 0.0278828  0.01492529 0.00919063 0.0063742  0.01525284 0.00330649\
 0.02869592 0.0191876  0.02351124 0.04465758 0.0175985  0.0056281\
 0.01753722 0.00303564 0.00975154 0.03427565 0.04038294 0.0107257"
 
''' str에서 숫자로 변환하는 구간 '''
fi_str = fi_str.split()
fi_float = [float(s) for s in fi_str]
print(fi_float)
fi_list = pd.Series(fi_float)

''' 25퍼 미만 인덱스 구하기 '''
low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
print('low_idx_list',low_idx_list)

''' 25퍼 미만 제거하기 '''
low_col_list = [x.columns[index] for index in low_idx_list]
# 이건 혹여 중복되는 값들이 많아 25퍼이상으로 넘어갈시 25퍼로 자르기
if len(low_col_list) > len(x.columns) * 0.25:   
    low_col_list = low_col_list[:int(len(x.columns)*0.25)]
print('low_col_list',low_col_list)
x.drop(low_col_list,axis=1,inplace=True)
print("after x.shape",x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from xgboost import XGBClassifier
model = XGBClassifier(cv=5, verbose=1,)

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


