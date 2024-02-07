import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC   # 옛날 머신



#1. 데이터
datasets = load_iris()
print(datasets)     # 0 1 2 카테고리 크로스 엔트로피
print(datasets.DESCR)   # 라벨 = 클래스
print(datasets.feature_names)   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = datasets.data
y = datasets.target
print(x.shape, y.shape)         # (150, 4) (150,) 회귀데이터 분류 데이터 헷갈릴수 있음 // 라벨 개수 한쪽으로 쏠리면 과적합 발생할수 있음
print(y)
print(np.unique(y, return_counts=True))     # (array([0, 1, 2]), array([50, 50, 50], dtype=int64))
print(pd.value_counts(y))           # y라벨 클래스 개수 확인 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, # y를 y_ohe3 고쳐도됨              
    train_size=0.8,
    random_state=1234,     
    stratify=y,     # 에러 : 분류에서만 쓴다. // y값이 정수로 딱 떨어지는것만 쓴다.
    shuffle=True,
    )
print(y_test)
print(np.unique(y_test, return_counts=True))        # (array([0, 1, 2]), array([10, 10, 10], dtype=int64))

#2. 모델 구성 
model = LinearSVC(C=100)

#.3 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
# results = model.evaluate(x_test, y_test)   
results = model.score(x_test, y_test)
print("model.score : ", results)  # acc
y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : " , acc)

# model.score :  0.8666666666666667
# [2 0 0 0 0 1 2 1 0 2 1 2 2 0 0 2 1 1 0 1 1 0 2 0 2 2 2 2 2 1]
# accuracy_score :  0.8666666666666667









