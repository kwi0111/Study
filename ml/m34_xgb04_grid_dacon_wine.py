#https://dacon.io/competitions/open/235610/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
import time
from sklearn.svm import SVR 
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV


#1. 데이터
path = "C:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv)

print(train_csv.shape) #(5497, 13)
print(test_csv.shape) #(1000, 12)
print(submission_csv.shape) #(1000, 2)


print(train_csv.columns) #'quality', 'fixed acidity', 'volatile acidity', 'citric acid',
    #    'residual sugar', 'chlorides', 'free sulfur dioxide',
    #    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
    #    'type'],
    
x = train_csv.drop(['quality'], axis= 1)
y = train_csv['quality']
y = y - 3

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0

test_csv.loc[test_csv['type'] == 'red', 'type'] = 1
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
# scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델구성
from xgboost import XGBClassifier
parameters = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.001, 0.0001],
    'max_depth': [3, 5, 7, 9, 11],
    'min_child_weight': [0, 0.01, 0.1, 1, 5, 10, 100],
    'subsample': [0.5, 0.7, 0.9, 1],
    'colsample_bytree': [0.5, 0.7, 0.9, 1],
    'gamma': [0, 1, 2, 3, 4, 5],
    'reg_alpha': [0, 0.01, 0.1, 1, 10],
    'reg_lambda': [0, 0.01, 0.1, 1, 10]
}
xgb = XGBClassifier(random_state=123)
model = RandomizedSearchCV(xgb, parameters, cv=kfold, random_state=123123,
                           n_jobs=22)
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

#4. 평가, 예측 
print("최상의 매개변수 : ", model.best_estimator_)  # 
print("최상의 매개변수 : ", model.best_params_) 
print("최상의 점수 : ", model.best_score_)  # 
results = model.score(x_test, y_test)
print("최고의 점수 : ", results)  


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))
print("걸린시간 :", round(end_time - start_time, 2), "초")


# 최상의 매개변수 :  {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 1}
# 최상의 점수 :  0.6095066708035992
# 최고의 점수 :  0.6136363636363636
# accuracy_score : 0.6136363636363636
# 걸린시간 : 2.37 초

# 최상의 매개변수 :  {'subsample': 0.9, 'reg_lambda': 0, 'reg_alpha': 1, 'n_estimators': 200, 'min_child_weight': 0.01, 'max_depth': 9, 'learning_rate': 0.001, 'gamma': 0, 'colsample_bytree': 0.7}        
# 최상의 점수 :  0.6097373047884994
# 최고의 점수 :  0.6245454545454545
# accuracy_score : 0.6245454545454545
# 걸린시간 : 8.06 초
