# https://dacon.io/competitions/open/235610/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras. callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import random
from sklearn.svm import LinearSVC

#1.데이터
path = "c:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)    # [5497 rows x 13 columns]
print(train_csv.shape)  # (5497, 13)
print(train_csv.head) 

test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
print(test_csv)     
print(test_csv.shape)  # (1000, 12)
print(test_csv.info()) 

submission_csv = pd.read_csv(path + 'sample_submission.csv')
print(submission_csv)
print(submission_csv.shape)  # (1000, 2)
# 결측치 처리 
print(train_csv.isna().sum()) # 없다
# print(np.unique(x, return_counts=True)) 
train_csv['type'] = train_csv['type'].map({"white":1, "red":0}).astype(int)
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0}).astype(int)
print(train_csv)
print(test_csv)     # [1000 rows x 12 columns]
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
model = LinearSVC(C=200)


#3.컴파일, 훈련
model.fit(x_train, y_train)


#4.평가, 예측
results = model.score(x_test, y_test) 
y_predict = model.predict(x_test)
 
acc = accuracy_score(y_test, y_predict)
print("acc : ", results)
'''
'''
# 로스 :  1.0740183591842651
# acc :  0.5515151619911194


'''
acc :  0.35818181818181816
'''

