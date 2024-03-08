
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
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
from sklearn.preprocessing import PolynomialFeatures # 다항 피쳐
import matplotlib.pyplot as plt




#1. 데이터
datasets = load_breast_cancer()

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
print('스테킹 ACC : ', accuracy_score(y_test, y_pred))

# model.socre :  0.9912280701754386
# 스테킹 ACC :  0.9912280701754386


plt.scatter(x[:, 0], y, color='blue', label='Original Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression Example')

# 다항식 회귀 그래프 그리기
x_plot = np.linspace(-1, 1, 100).reshape(-1, 1)
x_plot_poly = pf.transform(x_plot)
y_plot = model.predict(x_plot_poly)

plt.plot(x_plot, y_plot, color='red', label='Polynomial Regression')

plt.legend()
plt.show()












