import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import StackingClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures



#1. 데이터
path = "C:\\_data\\dacon\\dechul\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0 )
test_csv = pd.read_csv(path + "test.csv", index_col=0 )
submission_csv = pd.read_csv(path + "sample_submission.csv")

# 라벨 엔코더
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder() # 대출기간, 대출목적, 근로기간, 주택소유상태 // 라벨 인코더 : 카테고리형 피처를 숫자형으로 변환
train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = le.fit_transform(train_csv['대출목적'])
train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])

test_csv['주택소유상태'] = le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = le.fit_transform(test_csv['대출목적'])
test_csv['대출기간'] = test_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
test_csv['근로기간'] = le.fit_transform(test_csv['근로기간'])

train_csv['대출등급'] = le.fit_transform(train_csv['대출등급']) # 마지막에 와야함

# x와 y를 분리
x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

pf = PolynomialFeatures(
    degree=3,
    include_bias=False
)
x_poly = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_poly, y , random_state=777, train_size=0.8, stratify=y, shuffle=True
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
# model = LogisticRegression()
model = RandomForestClassifier()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.socre : ', model.score(x_test, y_test))
print('PF ACC : ', accuracy_score(y_test, y_pred))









##############################################
# 배깅 Ture acc_score :  0.5536363636363636
# 배깅 False acc_score :  0.5518181818181818

# 보팅 소프트 acc_score :  0.842151721273171
# 보팅 하드 acc_score :  0.7953164754140921

# 스테킹 cv = 3 acc_score :  0.8528480191079495
# 스테킹 cv = 5 acc_score :  0.8538864946258892

# PF ACC :  0.8381016667532063 랜포, degree 2
# PF ACC :  0.8306246430240407 랜포, degree 3 
'''
'''
