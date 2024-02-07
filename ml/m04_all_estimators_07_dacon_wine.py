# https://dacon.io/competitions/open/235610/mysubmission

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
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
allAlgorithms = all_estimators(type_filter='classifier')

for name, algorithm in allAlgorithms:
    try:
        #2. 모델
        model = algorithm()
        #.3 훈련
        model.fit(x_train, y_train)
        
        acc = model.score(x_test, y_test)   
        print(name, "의 정답률 : ", round(acc, 2))   
    except: 
        continue


'''
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

