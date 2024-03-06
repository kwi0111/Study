import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
x, y = fetch_covtype(return_X_y=True)
y = y - 1
x_train, x_test, y_train, y_test = train_test_split(
    x, y , random_state=777, train_size=0.8,
)
print(pd.value_counts(y_train))

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

########################## smote ############################### 데이터 증폭하는데 좋음
print("====================== smote 적용 =====================")
from imblearn.over_sampling import SMOTE
import sklearn as sk
print('사이킷런 : ', sk.__version__)    # 사이킷런 :  1.3.0

smote = SMOTE(random_state=123) # 랜덤 고정
x_train, y_train = smote.fit_resample(x_train, y_train) # 트레인 0.9 테스트 // 0.1은 그대로 (평가는 증폭 X)
print(pd.value_counts(y_train))

#2. 모델
model = RandomForestClassifier()

#.3 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score : ', acc)

# 랜포
# acc_score :  0.9549925561302204

# ====================== smote 적용 =====================
# 사이킷런 :  1.1.3
# 1    226994
# 6    226994
# 0    226994
# 2    226994
# 5    226994
# 4    226994
# 3    226994
# Name: count, dtype: int64
# 최종점수 :  0.9568513721676721
# acc_score :  0.9568513721676721






