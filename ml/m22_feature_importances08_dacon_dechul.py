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
print(train_csv.shape)  # (96294, 14)
test_csv = pd.read_csv(path + "test.csv", index_col=0 )
print(test_csv.shape)  # (64197, 13)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv.shape)  # (64197, 2)

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
# n_splits = 8
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델구성
from xgboost import XGBClassifier
model = XGBClassifier()

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

from sklearn.metrics import accuracy_score
print('score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))
print("최적튠 ACC :", accuracy_score(y_test, y_predict))
print("걸린시간 :", round(end_time - start_time, 2), "초")
print (type(model).__name__, model.feature_importances_)

# score : 0.8559560896009494
# accuracy_score : 0.8559560896009494
# 최적튠 ACC : 0.8559560896009494
# 걸린시간 : 0.79 초
# XGBClassifier [0.04624576 0.40506673 0.01161324 0.01615094 0.03441192 0.01665032
#  0.01378668 0.02518089 0.02123867 0.18819301 0.20061369 0.01066475
#  0.01018338]

# score : 0.8559560896009494
# accuracy_score : 0.8559560896009494
# 최적튠 ACC : 0.8559560896009494
# 걸린시간 : 0.8 초
# XGBClassifier [0.04624576 0.40506673 0.01161324 0.01615094 0.03441192 0.01665032
#  0.01378668 0.02518089 0.02123867 0.18819301 0.20061369 0.01066475
#  0.01018338]

# score : 0.8559560896009494
# accuracy_score : 0.8559560896009494
# 최적튠 ACC : 0.8559560896009494
# 걸린시간 : 0.78 초
# XGBClassifier [0.04624576 0.40506673 0.01161324 0.01615094 0.03441192 0.01665032
#  0.01378668 0.02518089 0.02123867 0.18819301 0.20061369 0.01066475
#  0.01018338]