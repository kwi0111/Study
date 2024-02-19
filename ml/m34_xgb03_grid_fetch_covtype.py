from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
from keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import tensorflow as tf
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV

#1. 데이터
datasets = fetch_covtype()

x = datasets.data
y = datasets.target
y = y - 1
#print(x.shape, y.shape) #(581012, 54) (581012,)
#print(pd.value_counts(y))
#print(np.unique(y, return_counts= True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],)
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123123)



#2. 모델구성
parameters = {
    'n_estimators': [100],
    'learning_rate': [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001],
    'max_depth': [3],
}

from xgboost import XGBClassifier
xgb = XGBClassifier(random_state=123)
model = RandomizedSearchCV(xgb, parameters, cv=kfold, random_state=123123,
                           n_jobs=22)

model = XGBClassifier(cv=5, verbose=1,)

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

#4. 평가, 예측 
results = model.score(x_test, y_test)
print("최고의 점수 : ", results)  


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))
print("걸린시간 :", round(end_time - start_time, 2), "초")


# 최고의 점수 :  0.8686264554271405
# accuracy_score : 0.8686264554271405
# 걸린시간 : 5.79 초