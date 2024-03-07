# 랜덤포레스트의 알고리즘이 배깅이다.
# 배깅 - 보팅
# votiong - 모델 여러개 / 같은 데이터 / 소프트 or 하드 / 하드는 다수결 / 소프트는 평균에서 높은쪽
# bagging - 모델 1개 - 데이터가 다르다 (샘플링해서) // 에포 느낌 = n이스터메이트 //  중복 된다 // 

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
'''
#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y , random_state=777, train_size=0.8,
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 3, # 트리 깊이
    'gamma' : 0,
    'min_child_weight' : 10,
    'min_child_weight' : 0,
    'subsample' : 0.4,
    'colsample_bytree' : 0.8,
    'colsample_bylevel' : 0.7,
    'colsample_bynode' : 1,
    'reg_alpha' : 0,
    'reg_lambda' : 1,
    'random_state' : 3377,
    'verbose' : 0,
}


#2. 모델
model = XGBClassifier()
model.set_params(early_stopping_rounds=10, **parameters)

#.3 훈련
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=1,
          eval_metric='logloss',
          )

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)
# 최종점수 :  0.9298245614035088
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score : ', acc)

##############################################
print(model.feature_importances_)

# for문을 사용해서 피처가 약한놈부터 하나씩 제거해서
# 30, 29, 28 ... 1까지

'''



'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# 데이터 로드
x, y = load_breast_cancer(return_X_y=True)

# 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=777, train_size=0.8,
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 10,  # 트리 깊이
    'gamma': 0,
    'min_child_weight': 10,
    'subsample': 0.4,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.7,
    'colsample_bynode': 1,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'random_state': 3377,
    'verbose': 0,
}

# 모델 정의 및 설정
model = XGBClassifier(**parameters)

# 피처 중요도 계산
model.fit(x_train, y_train)

feature_importances = model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]  # 각 피처의 중요도를 내림차순으로 정렬하여 인덱스를 얻습니다.

# 피처를 하나씩 제거하면서 모델 평가
scores = []
for i in range(len(sorted_indices), 0, -1):
    # 선택된 피처들의 인덱스
    selected_indices = sorted_indices[:i]
    
    # 선택된 피처로 데이터 재구성
    x_train_selected = x_train[:, selected_indices]
    x_test_selected = x_test[:, selected_indices]
    
    # 모델 재훈련
    model.fit(x_train_selected, y_train)
    
    # 모델 평가
    score = model.score(x_test_selected, y_test)
    scores.append((i, score))

# 결과 출력
for i, score in scores:
    print(f"선택된 피처 개수: {i}, 모델 성능: {score}")

'''


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression # 앤 분류다

datasets = load_breast_cancer()


x = datasets.data
y= datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size= 0.8)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Params = {'n_estimators' : 1000,
#           'learning_rate' : 0.01,
#           'max_depth' : 3,
#           'gamma' : 0,
#           'min_child_weight' : 0,
#           'subsample' : 0.4,
#           'colsample_bytree' : 0.8,
#           'colsample_lylevel' : 0.7,
#           'colsample_bynode' : 1,
#           'reg_alpha' : 0,
#           'reg_lamba' : 1,
#           'random_state' : 3377,
#           'verbose' : 0        
#           }

#2. 모델
model = BaggingClassifier(LogisticRegression(),
                          n_estimators=10,
                          n_jobs=20,
                          random_state=123,
                          bootstrap=True,   # 디폴트다 // 중복 허용 // False = 중복 허용 X
                          
                          )   # 랩핑

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score :', acc)

##########################
# 기존 xgb acc_score : 0.9473684210526315
# logisticR acc_score : 0.956140350877193


'''



'''
















