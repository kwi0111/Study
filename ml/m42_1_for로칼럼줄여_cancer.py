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
선택된 피처 개수: 30, 모델 성능: 0.9122807017543859
선택된 피처 개수: 29, 모델 성능: 0.9122807017543859
선택된 피처 개수: 28, 모델 성능: 0.9210526315789473
선택된 피처 개수: 27, 모델 성능: 0.9298245614035088
선택된 피처 개수: 26, 모델 성능: 0.9210526315789473
선택된 피처 개수: 25, 모델 성능: 0.9298245614035088
선택된 피처 개수: 24, 모델 성능: 0.9298245614035088
선택된 피처 개수: 23, 모델 성능: 0.9210526315789473
선택된 피처 개수: 22, 모델 성능: 0.9210526315789473
선택된 피처 개수: 21, 모델 성능: 0.9298245614035088
선택된 피처 개수: 20, 모델 성능: 0.9210526315789473
선택된 피처 개수: 19, 모델 성능: 0.9298245614035088
선택된 피처 개수: 18, 모델 성능: 0.9210526315789473
선택된 피처 개수: 17, 모델 성능: 0.9210526315789473
선택된 피처 개수: 16, 모델 성능: 0.9210526315789473
선택된 피처 개수: 15, 모델 성능: 0.9298245614035088
선택된 피처 개수: 14, 모델 성능: 0.9298245614035088
선택된 피처 개수: 13, 모델 성능: 0.9210526315789473
선택된 피처 개수: 12, 모델 성능: 0.9210526315789473
선택된 피처 개수: 11, 모델 성능: 0.9210526315789473
선택된 피처 개수: 10, 모델 성능: 0.9298245614035088
선택된 피처 개수: 9, 모델 성능: 0.9210526315789473
선택된 피처 개수: 8, 모델 성능: 0.9210526315789473
선택된 피처 개수: 7, 모델 성능: 0.9298245614035088
선택된 피처 개수: 6, 모델 성능: 0.9210526315789473
선택된 피처 개수: 5, 모델 성능: 0.9210526315789473
선택된 피처 개수: 4, 모델 성능: 0.9122807017543859
선택된 피처 개수: 3, 모델 성능: 0.9298245614035088
선택된 피처 개수: 2, 모델 성능: 0.8947368421052632
선택된 피처 개수: 1, 모델 성능: 0.8859649122807017


'''



