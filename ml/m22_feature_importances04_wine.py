from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC #softvector machine
from sklearn.linear_model import Perceptron, LogisticRegression , LinearRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


class CustomXGBClassifier(XGBClassifier):
    def __str__(self):
        return 'XGBClassifier()'

#1. 데이터

datasets = load_wine()
print(datasets.DESCR) 
print(datasets.feature_names) #'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 
'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'       


x= datasets.data
y= datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델구성

model1 = DecisionTreeClassifier(random_state=777)
model2 = RandomForestClassifier(random_state=777)
model3 = GradientBoostingClassifier(random_state=777)
model4 = CustomXGBClassifier(random_state=777, cv=kfold)
# model = CustomXGBClassifier(cv=5, verbose=1,)
models = [model1, model2, model3, model4]

for model in models :
    model.fit(x_train, y_train)
    print("========================", model, '=======================')
    print(model)
    print("acc : ", model.score(x_test, y_test))
    print(model.feature_importances_)

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


