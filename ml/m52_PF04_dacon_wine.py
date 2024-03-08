import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression # 앤 분류다
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import StackingClassifier, RandomForestClassifier


#1. 데이터
path = "C:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv)

print(train_csv.shape) #(5497, 13)
print(test_csv.shape) #(1000, 12)
print(submission_csv.shape) #(1000, 2)


print(train_csv.columns) #'quality', 'fixed acidity', 'volatile acidity', 'citric acid',
    #    'residual sugar', 'chlorides', 'free sulfur dioxide',
    #    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
    #    'type'],
    
x = train_csv.drop(['quality'], axis= 1)
y = train_csv['quality']
y = y - 3

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0

test_csv.loc[test_csv['type'] == 'red', 'type'] = 1
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

pf = PolynomialFeatures(
    degree=2,
    include_bias=False
)
x_poly = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_poly, y , random_state=777, train_size=0.8, stratify=y
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

# 보팅 하드 acc_score :  0.6772727272727272
# 보팅 소프트 acc_score :  0.6663636363636364

# 스테킹 acc_score :  0.6781818181818182

# PF ACC :  0.5645454545454546
# PF ACC :  0.6927272727272727 랜포
'''
Trech=0.055, n=12, ACC: 58.09%
Trech=0.064, n=11, ACC: 59.27%
Trech=0.067, n=10, ACC: 60.09%
Trech=0.069, n=9, ACC: 59.27%
Trech=0.070, n=8, ACC: 58.82%
Trech=0.074, n=7, ACC: 58.36%
Trech=0.074, n=6, ACC: 58.00%
Trech=0.077, n=5, ACC: 57.27%
Trech=0.092, n=4, ACC: 57.09%
Trech=0.097, n=3, ACC: 56.18%
Trech=0.114, n=2, ACC: 53.82%
Trech=0.149, n=1, ACC: 54.27%
'''
