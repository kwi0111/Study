# 주성분 분석 (Principal Component Analysis), PCA
# 복잡한 데이터를 차원 축소 알고리즘으로 조금 더 심플한 차원의 데이터로 만들어 분석 // 차원 축소

# 스케일링 후 PCA후 스플릿
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, KFold
import time
print(sk.__version__)

#1. 데이터
datasets = load_diabetes()


x = datasets['data']
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

# print(x)

x_train, x_test, y_train, y_test = train_test_split(
                                                    x,
                                                    y,             
                                                    train_size=0.86,
                                                    random_state=2024,
                                                    shuffle=True,
                                                    # stratify=y,
                                                    )
n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123123)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123123)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from xgboost import XGBRegressor
parameters = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.001, 0.0001],
    'max_depth': [3, 5, 7, 9, 11],
    'min_child_weight': [0, 0.01, 0.1, 1, 5, 10, 100],
    'subsample': [0.5, 0.7, 0.9, 1],
    'colsample_bytree': [0.5, 0.7, 0.9, 1],
    'gamma': [0, 1, 2, 3, 4, 5],
    'reg_alpha': [0, 0.01, 0.1, 1, 10],
    'reg_lambda': [0, 0.01, 0.1, 1, 10]
}
xgb = XGBRegressor(random_state=123)
model = RandomizedSearchCV(xgb, parameters, cv=kfold, random_state=123123,
                           n_jobs=22)

# 3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


# 4. 평가 예측
print('====================================')
print("최상의 매개변수 : ", model.best_estimator_) 
print("최상의 매개변수 : ", model.best_params_) 
print("최상의 점수 : ", model.best_score_)  
results = model.score(x_test, y_test)
print("최고의 점수 : ", results)  
print("걸린시간 :", round(end_time - start_time, 2), "초")




# [0.40242108 0.14923197 0.12059663 0.09554764 0.06621814 0.06027171
#  0.05365657 0.0433682  0.007832   0.00085607]

# for i in range(x.shape[1], 0, -1):
#     pca = PCA(n_components=i)
#     x_train = pca.fit_transform(x_train)
#     x_test = pca.transform(x_test)
    
#     # 모델 초기화
#     model = RandomForestClassifier()
    
#     # 모델 훈련
#     model.fit(x_train, y_train)
    
#     # 모델 평가
#     results = model.score(x_test, y_test)
#     print('====================================')
#     print(x_train.shape)
#     print('model.score : ', results)


# ====================================
# (442, 10) pca 안했따
# model.score :  0.511942510804726
# ====================================
# (442, 4)
# model.score :  0.46992522100698675
# ====================================
# (442, 8)
# model.score :  0.5251458328759518


# 최상의 매개변수 :  {'subsample': 0.9, 'reg_lambda': 1, 'reg_alpha': 1, 'n_estimators': 200, 'min_child_weight': 10, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 1, 'colsample_bytree': 1}
# 최상의 점수 :  0.41411401938147796
# 최고의 점수 :  0.46899581617090114
# 걸린시간 : 3.48 초
