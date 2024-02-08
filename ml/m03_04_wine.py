from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    train_size=0.8,
                                                    random_state=123,
                                                    stratify=y,
                                                    shuffle=True
                                                    )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# scaler = MinMaxScaler() # 클래스 정의
scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
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

'''
#3.컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("model.score : ", results)  # acc
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : " , acc)

# LinearSVC                     0.9444444444444444
# Perceptron                    1.0
# LogisticRegression            1.0
# KNeighborsClassifier          0.9444444444444444
# DecisionTreeClassifier         0.8888888888888888
# RandomForestClassifier        0.9722222222222222
'''


'''
for문
LinearSVC score :  0.94
LinearSVC predict :  0.91
Perceptron score :  1.0
Perceptron predict :  1.0
LogisticRegression score :  1.0
LogisticRegression predict :  1.0
KNeighborsClassifier score :  0.94
KNeighborsClassifier predict :  0.91
DecisionTreeClassifier score :  0.89
DecisionTreeClassifier predict :  0.82
RandomForestClassifier score :  0.97
RandomForestClassifier predict :  0.95

'''














 




