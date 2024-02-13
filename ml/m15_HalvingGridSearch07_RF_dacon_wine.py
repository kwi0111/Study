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
print(x)
y = train_csv['quality']



x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
print(x)

test_csv.loc[test_csv['type'] == 'red', 'type'] = 1
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

print(test_csv)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


n_splits=5
#kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
#n_split = 섞어서 분할하는 갯수
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델구성
parameters = [
    {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1, 2, 4], "min_samples_split": [2, 3, 5, 10]},
]    
     
model = HalvingGridSearchCV(RandomForestClassifier(), 
                     parameters,
                     cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1, 
                     random_state=66,
                     # n_iter=20,  # 디폴트 10
                     factor=5, # 디폴트 3 
                     min_resources=50,  # 데이터 조절하고싶으면 factor, min_resources 맞춘다.
                     )
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

from sklearn.metrics import accuracy_score
best_predict = model.best_estimator_.predict(x_test)
best_acc_score = accuracy_score(y_test, best_predict)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print('best_score :', model.best_score_)
print('score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적튠 ACC :", accuracy_score(y_test, y_predict))

print("걸린시간 :", round(end_time - start_time, 2), "초")



# 최적의 매개변수 :  RandomForestClassifier(n_jobs=2)
# 최적의 파라미터 :  {'min_samples_split': 2, 'n_jobs': 2}
# best_score : 0.6502184817457854
# score : 0.5236363636363637
# accuracy_score : 0.5236363636363637
# 최적튠 ACC : 0.5236363636363637
# 걸린시간 : 17.22 초

# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'min_samples_split': 3}
# best_score : 0.652721584445134
# score : 0.5227272727272727
# accuracy_score : 0.5227272727272727
# 최적튠 ACC : 0.5227272727272727
# 걸린시간 : 4.33 초


# n_iterations: 3
# n_required_iterations: 3
# n_possible_iterations: 3
# min_resources_: 50
# max_resources_: 4397
# aggressive_elimination: False
# factor: 5
# ----------
# iter: 0
# n_candidates: 60
# n_resources: 50
# Fitting 5 folds for each of 60 candidates, totalling 300 fits
# C:\Users\AIA\anaconda3\Lib\site-packages\sklearn\model_selection\_split.py:684: UserWarning: The least populated class in y has only 4 members, which is less than n_splits=5.
#   warnings.warn(
# ----------
# iter: 1
# n_candidates: 12
# n_resources: 250
# Fitting 5 folds for each of 12 candidates, totalling 60 fits
# C:\Users\AIA\anaconda3\Lib\site-packages\sklearn\model_selection\_split.py:684: UserWarning: The least populated class in y has only 4 members, which is less than n_splits=5.
#   warnings.warn(
# ----------
# iter: 2
# n_candidates: 3
# n_resources: 1250
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# C:\Users\AIA\anaconda3\Lib\site-packages\sklearn\model_selection\_split.py:684: UserWarning: The least populated class in y has only 4 members, which is less than n_splits=5.
#   warnings.warn(
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3)
# 최적의 파라미터 :  {'min_samples_split': 3}
# best_score : 0.5869879518072288
# score : 0.5345454545454545
# accuracy_score : 0.5345454545454545
# 최적튠 ACC : 0.5345454545454545
# 걸린시간 : 4.58 초