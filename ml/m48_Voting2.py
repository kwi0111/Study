
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
from sklearn.linear_model import LogisticRegression # 앤 분류다

#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y= datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size= 0.8,  stratify=y)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

model = VotingClassifier(
    estimators=[('LR', lr), ('RF', rf), ('XGB', xgb)],
    # voting='soft',
    voting='hard',  # 디폴트
    
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
# 기존 xgb acc_score : 0.9473684210526315
# logisticR acc_score : 0.956140350877193

# 최종점수 0.9824561403508771
# acc_score : 0.9824561403508771
'''



'''
model_class = [xgb, rf, lr]
for model2 in model_class:
    model2.fit(x_train, y_train)
    y_pred = model2.predict(x_test)
    score2=accuracy_score(y_test, y_pred)
    class_name = model2.__class__.__name__
    print("{0} 정확도 : {1:.4f}".format(class_name,score2))














