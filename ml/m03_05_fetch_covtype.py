from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    train_size=0.9,
                                                    random_state=123,
                                                    stratify=y,
                                                    shuffle=True
                                                    ) 

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

#3.컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test) 
y_predict = model.predict(x_test)
 
acc = accuracy_score(y_test, y_predict)
print("acc : ", results)

# LinearSVC                     
# Perceptron                  acc :  0.5297580117724002
# LogisticRegression           acc :  0.6179477470655055
# KNeighborsClassifier        acc :  0.9706722660149393
# DecisionTreeClassifier      acc :  0.9428246876183264
# RandomForestClassifier        acc :  0.9572648101614403


'''

'''