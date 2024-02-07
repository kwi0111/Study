from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

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
model = LinearSVC(C=1000)


#3.컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test) 
y_predict = model.predict(x_test)
 
acc = accuracy_score(y_test, y_predict)
print("acc : ", results)

# accuracy_score :  0.7971825168024922

# 그냥
# 로스 : 0.6306923627853394

# MinMaxScaler
# 로스 :  0.6288957595825195

# StandardScaler
# 로스 :  0.628129780292511

# MaxAbsScaler
# 로스 :  

# RobustScaler
# 로스 :  