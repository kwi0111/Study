import numpy as np
from sklearn.datasets import load_iris    # 유방암
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC, LinearSVR # SVC는 분류형 모델, SVR은 회귀형(예측) 모델
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터 
x,y = load_iris(return_X_y=True) 

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    random_state=123,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    )

#2. 모델 구성
# model = LinearSVC(C=100)
# model = Perceptron()
# model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()

# models = [LinearSVC(),Perceptron(),LogisticRegression(),KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier()]
# ############## 훈련 반복 for 문 ###################
# for model in models :
#     try:
#         model.fit(x_train, y_train)
#         result = model.score(x_test, y_test)
#         print(f'{type(model).__name__} score : ', round(result, 2))
        
#         y_predict = model.predict(x_test)
#         print(f'{type(model).__name__} predict : ', round(r2_score(y_test,y_predict), 2))
#     except:
#         continue

#.3 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("model.score : ", results)  # acc
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : " , round(acc, 2))

# LinearSVC                     0.9210526315789473
# Perceptron                  0.8859649122807017
# LogisticRegression           0.9824561403508771
# KNeighborsClassifier        0.9649122807017544
# DecisionTreeClassifier      0.9649122807017544
# RandomForestClassifier        0.9912280701754386


'''
for문
LinearSVC score :  0.97
LinearSVC predict :  0.96
Perceptron score :  0.87
Perceptron predict :  0.83
LogisticRegression score :  1.0
LogisticRegression predict :  1.0
KNeighborsClassifier score :  0.97
KNeighborsClassifier predict :  0.96
DecisionTreeClassifier score :  0.93
DecisionTreeClassifier predict :  0.92
RandomForestClassifier score :  0.93
RandomForestClassifier predict :  0.92
'''