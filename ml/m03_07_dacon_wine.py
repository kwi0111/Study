# https://dacon.io/competitions/open/235610/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras. callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, r2_score
import random
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings(action='ignore')

#1.데이터
path = "c:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 결측치 처리 
train_csv['type'] = train_csv['type'].map({"white":1, "red":0}).astype(int)
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0}).astype(int)

# x와 y를 분리
x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']

x_train, x_test, y_train, y_test = train_test_split(
x, y,             
train_size=0.7,
random_state=123,
stratify=y,  
shuffle=True,
)

#2. 모델 구성 
models = [LinearSVC(),Perceptron(),LogisticRegression(),KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier()]
############## 훈련 반복 for 문 ###################
for model in models :
    try:
        model.fit(x_train, y_train)
        result = model.score(x_test, y_test)
        print(f'{type(model).__name__} score : ', round(result, 2))
        
        y_predict = model.predict(x_test)
        print(f'{type(model).__name__} predict : ', round(r2_score(y_test,y_predict), 2))
    except:
        continue

#3.컴파일, 훈련
model.fit(x_train, y_train)


#4.평가, 예측
results = model.score(x_test, y_test) 
y_predict = model.predict(x_test)
 
acc = accuracy_score(y_test, y_predict)
print("acc : ", results)


'''
# LinearSVC                    0.44303030303030305 
# Perceptron                 0.3321212121212121
# LogisticRegression           0.4915151515151515
# KNeighborsClassifier        0.4696969696969697
# DecisionTreeClassifier      0.5793939393939394
# RandomForestClassifier        0.6642424242424242
'''

'''
LinearSVC score :  0.34
LinearSVC predict :  -0.78
Perceptron score :  0.33
Perceptron predict :  -0.96
LogisticRegression score :  0.49
LogisticRegression predict :  -0.01
KNeighborsClassifier score :  0.47
KNeighborsClassifier predict :  -0.2
DecisionTreeClassifier score :  0.59
DecisionTreeClassifier predict :  0.02
RandomForestClassifier score :  0.66
RandomForestClassifier predict :  0.36
'''

