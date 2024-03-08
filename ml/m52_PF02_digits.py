import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression # 앤 분류다
from sklearn.preprocessing import PolynomialFeatures # 다항 피쳐


#1.
datasets = load_digits()


x = datasets.data
y= datasets.target


pf = PolynomialFeatures(
    degree=3,
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
# acc_score : 0.9555555555555556

# 보팅 하드 acc_score : 0.975
# 보팅 소프트 acc_score : acc_score : 0.9805555555555555

# 스테킹 acc_score : 0.9777777777777777

# degree 2 PF ACC :  0.9805555555555555
# degree 3 PF ACC :  0.9833333333333333
'''



'''
















