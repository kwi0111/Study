
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
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

# 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=777, train_size=0.8,
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 10,  # 트리 깊이
    'gamma': 0,
    'min_child_weight': 10,
    'subsample': 0.4,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.7,
    'colsample_bynode': 1,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'random_state': 3377,
    'verbose': 0,
}

# 모델 정의 및 설정
model = XGBClassifier(**parameters)

# 피처 중요도 계산
model.fit(x_train, y_train)

feature_importances = model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]  # 각 피처의 중요도를 내림차순으로 정렬하여 인덱스를 얻습니다.

# 피처를 하나씩 제거하면서 모델 평가
scores = []
for i in range(len(sorted_indices), 0, -1):
    # 선택된 피처들의 인덱스
    selected_indices = sorted_indices[:i]
    
    # 선택된 피처로 데이터 재구성
    x_train_selected = x_train[:, selected_indices]
    x_test_selected = x_test[:, selected_indices]
    
    # 모델 재훈련
    model.fit(x_train_selected, y_train)
    
    # 모델 평가
    score = model.score(x_test_selected, y_test)
    scores.append((i, score))

# 결과 출력
for i, score in scores:
    print(f"선택된 피처 개수: {i}, 모델 성능: {score}")


# 선택된 피처 개수: 13, 모델 성능: 0.8392439898229399
# 선택된 피처 개수: 12, 모델 성능: 0.8382055143050002
# 선택된 피처 개수: 11, 모델 성능: 0.8363362583727089
# 선택된 피처 개수: 10, 모델 성능: 0.839036294719352
# 선택된 피처 개수: 9, 모델 성능: 0.8399709226854977
# 선택된 피처 개수: 8, 모델 성능: 0.8431382730152137
# 선택된 피처 개수: 7, 모델 성능: 0.8398151513578067
# 선택된 피처 개수: 6, 모델 성능: 0.8380497429773093
# 선택된 피처 개수: 5, 모델 성능: 0.8443844436367413
# 선택된 피처 개수: 4, 모델 성능: 0.8535749519705073
# 선택된 피처 개수: 3, 모델 성능: 0.8120878550288178
# 선택된 피처 개수: 2, 모델 성능: 0.37416272911366116
# 선택된 피처 개수: 1, 모델 성능: 0.34918739290721224






