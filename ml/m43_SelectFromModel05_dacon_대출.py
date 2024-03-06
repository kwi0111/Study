import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


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
    x, y , random_state=777, train_size=0.8,
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 3, # 트리 깊이
    'gamma' : 0,
    'min_child_weight' : 10,
    'min_child_weight' : 0,
    'subsample' : 0.4,
    'colsample_bytree' : 0.8,
    'colsample_bylevel' : 0.7,
    'colsample_bynode' : 1,
    'reg_alpha' : 0,
    'reg_lambda' : 1,
    'random_state' : 3377,
    'verbose' : 0,
}


#2. 모델
model = XGBClassifier()
model.set_params(early_stopping_rounds=10, **parameters)

#.3 훈련
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=1,
          eval_metric='mlogloss',
          )

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)
# 최종점수 :  0.9298245614035088
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score : ', acc)

##############################################
print(model.feature_importances_)

thresholds = np.sort(model.feature_importances_)    # 내림차순
# print(thresholds)
# [0.01226016 0.01844304 0.01364688 0.04408791 0.01009422 0.01062047
#  0.03175311 0.06426384 0.00957733 0.01629062 0.01834335 0.01561584
#  0.01365232 0.03140673 0.01297489 0.01083888 0.01846627 0.01291327
#  0.01083225 0.01474823 0.11274869 0.02523725 0.13143815 0.10594388
#  0.01828141 0.02078197 0.04221498 0.11370341 0.02124572 0.0175748 ]
# [0.00957733 0.01009422 0.01062047 0.01083225 0.01083888 0.01226016
#  0.01291327 0.01297489 0.01364688 0.01365232 0.01474823 0.01561584
#  0.01629062 0.0175748  0.01828141 0.01834335 0.01844304 0.01846627
#  0.02078197 0.02124572 0.02523725 0.03140673 0.03175311 0.04221498
#  0.04408791 0.06426384 0.10594388 0.11274869 0.11370341 0.13143815]
from sklearn.feature_selection import SelectFromModel # 크거나 같은값의 피처는 삭제해버린다.
print("="*100)
for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)   # 클래스를 인스턴스화 한다 // 
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(i, "\t변형된 x_train: ", select_x_train.shape, "변형된 x_test: ", select_x_test.shape )
    
    select_model =XGBClassifier()
    select_model.set_params(
        early_stopping_rounds=10,
        **parameters,
        eval_metric = 'mlogloss',
        
    )
    select_model.fit(select_x_train, y_train,
                     eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
                     verbose=0,
                     )
    select_y_predict = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_predict)
    print("Trech=%.3f, n=%d, ACC: %.2f%%" %(i, select_x_train.shape[1], score*100))

'''
Trech=0.006, n=13, ACC: 65.61%
Trech=0.008, n=12, ACC: 64.70%
Trech=0.013, n=11, ACC: 63.54%
Trech=0.024, n=10, ACC: 65.02%
Trech=0.040, n=9, ACC: 63.59%
Trech=0.042, n=8, ACC: 64.89%
Trech=0.047, n=7, ACC: 62.94%
Trech=0.058, n=6, ACC: 59.85%
Trech=0.063, n=5, ACC: 63.48%
Trech=0.071, n=4, ACC: 66.36%
Trech=0.177, n=3, ACC: 59.42%
Trech=0.201, n=2, ACC: 36.53%
Trech=0.250, n=1, ACC: 34.92%
'''
