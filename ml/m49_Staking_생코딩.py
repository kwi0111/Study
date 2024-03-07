
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
from catboost import CatBoostClassifier

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

li = []
li2 = []

model_class = [xgb, rf, lr]
for model in model_class:
    model.fit(x_train, y_train)
    y_pred1 = model.predict(x_train)
    # print(y_pred.shape) # (114,) 백터형태로 출력
    y_pred_test = model.predict(x_test)
    li.append(y_pred1)
    li2.append(y_pred_test)

    score = accuracy_score(y_test, y_pred_test)
    class_name = model.__class__.__name__
    print("{0} ACC : {1:.4f}".format(class_name,score))
# print(li)   # 리스트는 shape X

new_x_train = np.array(li).T
new_x_test = np.array(li2).T
# print(y_data, y_data.shape) # (114, 3)
# print(new_x_train, new_x_train.shape)  # (455, 3)
# print(new_x_test, new_x_test.shape)  # (114, 3)
# print(y_train.shape, y_test.shape)  # (455,) (114,)


# 이제 두번째 모델 ㄱㄱ // 과적합만 조심하면 쓸만함
model2 = CatBoostClassifier(verbose=0)
model2.fit(new_x_train, y_train)
y_pred2 = model2.predict(new_x_test)
# score2 = accuracy_score(y_staking_data, y_test)
score2 = accuracy_score(y_test, y_pred2)
print("스테킹 결과 : ", score2)



