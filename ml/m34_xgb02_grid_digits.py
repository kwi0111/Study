from sklearn.datasets import load_digits
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time

datasets = load_digits()

x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape) # (1797, 64) (1797,)
print(pd.value_counts(y, sort=False))
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        train_size=0.7,
        random_state=200,    
        stratify=y,
        shuffle=True,
        )
scaler = MinMaxScaler()
# scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

from xgboost import XGBClassifier

#2. 모델구성
parameters = {
    'n_estimators': [100],
    'learning_rate': [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001],
    'max_depth': [3],
}

#2.모델
xgb = XGBClassifier(random_state=123)
model = RandomizedSearchCV(xgb, parameters, cv=kfold, random_state=123123,
                           n_jobs=22)

#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

#4. 평가, 예측 
print("최상의 매개변수 : ", model.best_estimator_)  # 
print("최상의 매개변수 : ", model.best_params_) 
print("최상의 점수 : ", model.best_score_)  # 
results = model.score(x_test, y_test)
print("최고의 점수 : ", results)  


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))
print("걸린시간 :", round(end_time - start_time, 2), "초")

# 최상의 점수 :  0.9562606715993169
# 최고의 점수 :  0.9722222222222222
# accuracy_score : 0.9722222222222222
# 걸린시간 : 2.39 초