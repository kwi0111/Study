import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings(action='ignore')
import time
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV


#1. 데이터
path = "C:\\_data\\dacon\\dechul\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0 )
test_csv = pd.read_csv(path + "test.csv", index_col=0 )
submission_csv = pd.read_csv(path + "sample_submission.csv")

# 라벨 엔코더
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder() # 대출기간, 대출목적, 근로기간, 주택소유상태 // 라벨 인코더 : 카테고리형 피처를 숫자형으로 변환
train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = le.fit_transform(train_csv['대출목적'])
train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])

test_csv['주택소유상태'] = le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = le.fit_transform(test_csv['대출목적'])
test_csv['대출기간'] = test_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
test_csv['근로기간'] = le.fit_transform(test_csv['근로기간'])

train_csv['대출등급'] = le.fit_transform(train_csv['대출등급']) # 마지막에 와야함

# x와 y를 분리
x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']


x_train, x_test, y_train, y_test = train_test_split(
                                                    x,
                                                    y,             
                                                    train_size=0.86,
                                                    random_state=2024,
                                                    stratify=y,
                                                    shuffle=True,
                                                    )
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from xgboost import XGBClassifier
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
xgb = XGBClassifier(random_state=123)
model = RandomizedSearchCV(xgb, parameters, cv=kfold, random_state=123123,
                           n_jobs=22)

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

#4. 평가, 예측 
print("최상의 매개변수 : ", model.best_estimator_) 
print("최상의 매개변수 : ", model.best_params_) 
print("최상의 점수 : ", model.best_score_)  
results = model.score(x_test, y_test)
print("최고의 점수 : ", results)  


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))
print("걸린시간 :", round(end_time - start_time, 2), "초")

# 최상의 매개변수 :  {'subsample': 0.7, 'reg_lambda': 0, 'reg_alpha': 0.01, 'n_estimators': 400, 'min_child_weight': 0.1, 'max_depth': 11, 'learning_rate': 0.01, 'gamma': 4, 'colsample_bytree': 1}  
# 최상의 점수 :  0.8464594567486422
# 최고의 점수 :  0.852618305889334
# accuracy_score : 0.852618305889334
# 걸린시간 : 118.58 초