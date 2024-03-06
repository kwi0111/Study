# m11_1 카피

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 18], dtype=int64))
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
print(y)
print('---------------------------------------------------------')

x = x[:-35]
y = y[:-35]      # # 0을 줄여버리겠다.
print(x)
print(y)
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    13

# 불균형하게 데이터 만들어놨다 // 증폭하기 위해서
############################################ 신뢰..? // 너무 많이 떨어져있는 아이들은 ACC보다 F1스코어로 봐야함
for i, v in enumerate(y):
    if v <=0:
        y[i] = 0
    else:
        y[i] = 1

print(pd.value_counts(y))
# 1    84
# 0    59

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=123,
    stratify=y
)

from keras.models import Sequential
from keras.layers import Dense
 
########################## smote ############################### 데이터 증폭하는데 좋음
print("====================== smote 적용 =====================")
from imblearn.over_sampling import SMOTE
import sklearn as sk
print('사이킷런 : ', sk.__version__)    # 사이킷런 :  1.3.0

smote = SMOTE(random_state=123) # 랜덤 고정
x_train, y_train = smote.fit_resample(x_train, y_train) # 트레인 0.9 테스트 // 0.1은 그대로 (평가는 증폭 X)
print(pd.value_counts(y_train))
# 0    75
# 1    75

#2.모델
model = RandomForestClassifier()
model.fit(x_train, y_train)

#4.평가, 예측
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average='macro')

print("정확도:", acc)
print("F1 스코어:", f1)


# 정확도: 0.9333333333333333
# F1 스코어: 0.9321266968325792





