from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , accuracy_score
from keras.models import Sequential , load_model
from keras.layers import Dense
import numpy as np
import time
import matplotlib.pyplot as plt

#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape,y.shape)          # (442, 10) (442,)
print(datasets.feature_names)   #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state= 151235 , shuffle= True )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2 모델구성
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.linear_model import Perceptron , LogisticRegression , LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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



#3 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy' , 'mse', 'mae'])     
# binary_crossentropy = 2진 분류 = y 가 2개면 무조건 이걸 사용한다
# accuracy = 정확도 = acc
# metrics로 훈련되는걸 볼 수는 있지만 가중치에 영향을 미치진 않는다.

# metrics가 accuracy 

#4. 평가, 예측
# results = model.score(x_test, y_test) 
# y_predict = model.predict(x_test) 
# print("acc : ", results)
# print("걸린시간 : ", round(end_time - start_time, 2),"초")  
# from sklearn.metrics import r2_score  
# r2 = r2_score(y_test, y_predict)                                                # 실제값, 예측값 순서
# print("r2 스코어 : " , r2)

'''
for문
LinearSVC score :  0.0
LinearSVC predict :  0.02
Perceptron score :  0.0
Perceptron predict :  -0.42
LogisticRegression score :  0.0
LogisticRegression predict :  0.06
KNeighborsClassifier score :  0.0
KNeighborsClassifier predict :  -0.76
DecisionTreeClassifier score :  0.01
DecisionTreeClassifier predict :  0.02
RandomForestClassifier score :  0.01
RandomForestClassifier predict :  -0.07
'''

