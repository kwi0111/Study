# 랜덤포레스트의 알고리즘이 배깅이다.
# 배깅 - 보팅
# votiong - 모델 여러개 / 같은 데이터 / 소프트 or 하드 / 하드는 다수결 / 소프트는 평균에서 높은쪽
# bagging - 모델 1개 - 데이터가 다르다 (샘플링해서) // 에포 느낌 = n이스터메이트 //  중복 된다 // 

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression # 앤 분류다
from sklearn.ensemble import StackingRegressor
from catboost import CatBoostRegressor

#1. 데이터
datasets = fetch_california_housing()
x = np.array(datasets.data)
y = np.array(datasets.target)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size= 0.8)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBRegressor(learning_rate=0.1, max_depth=3, min_child_weight=1, n_estimators=100, random_state=777)
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=777)
lr = LinearRegression()

model = StackingRegressor(
    estimators=[('XGB', xgb), ('RF', rf), ('LR', lr)],
    final_estimator = CatBoostRegressor(verbose=0),
    n_jobs=-1,
    cv = 5,
    passthrough=True,
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수', results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2_score :', r2)

##########################

# 리니어 Ture r2_score : 0.5827784121245206
# 랜포 True 최종점수 0.8021630262259856

# 보팅 regressor r2_score : 0.7559852381903986

# 스테킹 cv3 r2_score : 0.8465688911293758
# 스케팅 cv5 r2_score : 0.8469958475025819
'''



'''
















