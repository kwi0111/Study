import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


#1. 데이터 

#1. 데이터 // 판다스, 넘파이 
path = "C:\\_data\\dacon\\diabetes\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)    
test_csv = pd.read_csv(path + "test.csv", index_col=0)      # 헤더는 기본 첫번째 줄이 디폴트값
submission_csv = pd.read_csv(path + "sample_submission.csv")       

############# x 와 y를 분리 ################
x = train_csv.drop(['Outcome', 'Insulin'], axis=1)   # 행삭제 : axis = 0 // 열삭제 : axis = 1 // train_csv에 있는 'Outcome'열 삭제
print(x)
y = train_csv.drop(['Insulin'], axis=1)       # 행삭제 : axis = 0 // 열삭제 : axis = 1 // train_csv에 있는 'Outcome'열 삭제 
y = train_csv['Outcome']                      # train_csv에 있는 'Outcome'열을 y로 설정
test_csv = test_csv.drop(['Insulin'], axis=1)
print(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.9, random_state=123123,
)


#2. 모델 구성
model = LinearSVC(C=100)

#.3 컴파일 훈련 // 하나 포기하는게 과적합 안걸림 // 
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("model.score : ", results)  


# model.score :  0.5606060606060606