import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score
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
############################################ 신뢰..? // 너무 많이 떨어져있는 아이들은 ACC보다 F1스코어로 봐야함
for i, v in enumerate(y):
    if v <= 4:  # 0, 1, 2, 3, 4 클래스를 0으로 변환
        y[i] = 0
    elif v == 5:  # 5 클래스를 1로 변환
        y[i] = 1
    elif v == 6:  # 6 클래스를 2로 변환
        y[i] = 2
    elif v == 7:  # 7 클래스를 3로 변환
        y[i] = 3
    else:  # 8, 9 클래스를 4로 변환
        y[i] = 4
        
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

########################## smote ############################### 데이터 증폭하는데 좋음
print("====================== smote 적용 =====================")
from imblearn.over_sampling import SMOTE
import sklearn as sk
print('사이킷런 : ', sk.__version__)    # 사이킷런 :  1.3.0

smote = SMOTE(random_state=123) # 랜덤 고정
x_train, y_train = smote.fit_resample(x_train, y_train) # 트레인 0.9 테스트 // 0.1은 그대로 (평가는 증폭 X)
print(pd.value_counts(y_train))

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
# 최종점수 :  0.9298245614035088
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score : ', acc)
f1 = f1_score(y_test, y_predict, average='macro')
print('f1_score : ', f1)


'''
acc_score :  0.6561088322342801
f1_score :  0.5110709735465466
====================== smote 적용 =====================
사이킷런 :  1.1.3
대출등급
2    22958
0    22958
3    22958
4    22958
1    22958
5    22958
6    22958
acc_score :  0.557401734254115
f1_score :  0.48181558644379424
'''







