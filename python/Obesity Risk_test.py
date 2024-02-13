import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from catboost import CatBoostClassifier
import datetime

# 데이터 로드
path = 'C:\\_data\\kaggle\\Obesity_Risk\\'
train=pd.read_csv(path+"train.csv",index_col=0)
test=pd.read_csv(path+"test.csv",index_col=0)
submission_csv=pd.read_csv(path+"sample_submission.csv")
x = train.drop(['NObeyesdad'], axis=1)
y = train['NObeyesdad']

# 훈련 데이터와 테스트 데이터 합치기
combined_data = pd.concat([x, test])

# 라벨 인코딩
lb = LabelEncoder()
columns_to_encode = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']
for column in columns_to_encode:
    lb.fit(combined_data[column])
    x[column] = lb.transform(x[column])
    test[column] = lb.transform(test[column])

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.90, random_state=3, stratify=y)

# 데이터 스케일링
scaler = StandardScaler() 
# scaler = MaxAbsScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler() 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test = scaler.transform(test)

# CatBoost 모델 정의
random_state = np.random.randint(1, 100)
CatBoost_params = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "verbose": False,
    "random_state": random_state,
    "learning_rate": 0.03,                # 학습률
    "n_estimators": 1500,                  # 부스팅 라운드 수
    "colsample_bylevel": 0.7,
    "min_child_samples": 10,
    "bootstrap_type": "Bernoulli",         # 부트스트랩 유형
    "subsample": 0.8,                      # 샘플 선택 비율
    "depth": 8,                            # 트리의 최대 깊이
    "border_count": 128,                   # 범주형 변수 수에 대한 근사값
    "l2_leaf_reg": 5,                      # L2 정규화 강도
    "leaf_estimation_iterations": 20,      # 각 잎 노드의 초기 추정 횟수
    "leaf_estimation_method": "Gradient",  # 잎 노드의 초기 추정 방법
    "thread_count": -1                     # 사용할 스레드 수 (-1은 가능한 모든 스레드 사용)
}






model = CatBoostClassifier(**CatBoost_params)

# 모델 학습
model.fit(x_train, y_train)

# 테스트 데이터 예측
y_pred = model.predict(x_test)
y_submit = model.predict(test)

# 정확도 계산
accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy)

# submission_csv 업데이트
submission_csv['NObeyesdad'] = pd.Series(y_submit.ravel())

# 결과 저장
dt = datetime.datetime.now()
submission_csv.to_csv(path + f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_acc_{accuracy:.4f}.csv", index=False)
