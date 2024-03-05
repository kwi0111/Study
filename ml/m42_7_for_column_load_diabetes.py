
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor

#1. 데이터
datasets = load_diabetes()


x = datasets['data']
y = datasets.target

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
model = XGBRegressor(**parameters)

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


# 선택된 피처 개수: 10, 모델 성능: 0.4104799210694783
# 선택된 피처 개수: 9, 모델 성능: 0.4038266792360645
# 선택된 피처 개수: 8, 모델 성능: 0.4455882479009936
# 선택된 피처 개수: 7, 모델 성능: 0.4116530585546333
# 선택된 피처 개수: 6, 모델 성능: 0.42474293010816067
# 선택된 피처 개수: 5, 모델 성능: 0.39462980869107134
# 선택된 피처 개수: 4, 모델 성능: 0.42408155110619394
# 선택된 피처 개수: 3, 모델 성능: 0.4191088313756042
# 선택된 피처 개수: 2, 모델 성능: 0.4370379582726256
# 선택된 피처 개수: 1, 모델 성능: 0.1747052637690072





