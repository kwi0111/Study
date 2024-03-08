
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression # 앤 분류다
from sklearn.preprocessing import PolynomialFeatures

datasets = fetch_covtype()


x = datasets.data
y= datasets.target


pf = PolynomialFeatures(
    degree=2,
    include_bias=False
)
x_poly = pf.fit_transform(x)
# print(x_poly)


x_poly, x_test, y_train, y_test = train_test_split(x_poly, y, random_state=777, train_size= 0.8,  stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_poly)
x_test = scaler.transform(x_test)



#2. 모델
model = LogisticRegression()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.socre : ', model.score(x_test, y_test))
print('PF ACC : ', accuracy_score(y_test, y_pred))

##########################
# 배깅 True acc_score : 0.7186905673691729 -> n_estimators=100으로 acc_score : 0.718707778628779
# 배깅 False acc_score : 0.7186991729989759

# 보팅 하드 acc_score : 0.8842800960388286
# 보팅 소프트 acc_score : 0.9016634682409232

# 스테킹 cv = 3 acc_score : 0.9625999328760876
# 스테킹 cv = 5 acc_score : 0.9622126795349518

# PF ACC :  0.7470375119403113 리니어
# PF ACC :  0.9568599777974751 랜포

'''



'''
















