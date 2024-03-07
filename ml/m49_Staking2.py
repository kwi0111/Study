
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from catboost import CatBoostClassifier

#1. 데이터
datasets = load_breast_cancer()

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

models = [xgb, rf, lr]
model_names = ['XGBClassifier', 'RandomForestClassifier', 'LogisticRegression']
results = []

# 개별 모델 학습 및 평가
for model, name in zip(models, model_names):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append(y_pred)
    print(f"{name} 정확도 : {accuracy:.4f}")

# 스태킹
stacked = np.array(results)
stacked = np.transpose(stacked)
meta_model = CatBoostClassifier()
meta_model.fit(stacked, y_test)

# 스태킹된 모델 평가
stacked_pred = meta_model.predict(stacked)
stacked_accuracy = accuracy_score(y_test, stacked_pred)
print(f"\nXGBClassifier 정확도 : {accuracy_score(y_test, results[0]):.4f}")
print(f"RandomForestClassifier 정확도 : {accuracy_score(y_test, results[1]):.4f}")
print(f"LogisticRegression 정확도 : {accuracy_score(y_test, results[2]):.4f}")
print(f"Stacking Test Accuracy: {stacked_accuracy:.4f}")





