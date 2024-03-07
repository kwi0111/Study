# 랜덤포레스트의 알고리즘이 배깅이다.
# 배깅 - 보팅
# votiong - 모델 여러개 / 같은 데이터 / 소프트 or 하드 / 하드는 다수결 / 소프트는 평균에서 높은쪽
# bagging - 모델 1개 - 데이터가 다르다 (샘플링해서) // 에포 느낌 = n이스터메이트 //  중복 된다 // 
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression # 앤 분류다
from catboost import CatBoostClassifier

datasets = fetch_covtype()


x = datasets.data
y= datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size= 0.8, stratify=y)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

model = StackingClassifier(
    estimators=[('XGB', xgb), ('RF', rf), ('LR', lr)],
    final_estimator = CatBoostClassifier(verbose=0),
    n_jobs=-1,
    cv = 3,
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score :', acc)

##########################
# 배깅 True acc_score : 0.7186905673691729 -> n_estimators=100으로 acc_score : 0.718707778628779
# 배깅 False acc_score : 0.7186991729989759

# 보팅 하드 acc_score : 0.8842800960388286
# 보팅 소프트 acc_score : 0.9016634682409232

# 스테킹 cv = 3 acc_score : 0.9625999328760876
# 스테킹 cv = 5 acc_score : 0.9622126795349518
'''



'''
















