from sklearn.datasets import load_digits
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time

#1.데이터
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
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
scaler = MinMaxScaler() # 클래스 정의
# scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
parameters = [
    {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]}, # 12
    {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]}, # 16
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]}, # 16
    {"min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1,], "min_samples_split": [2, 3, 5, 10]}, # 
]    

from sklearn.pipeline import make_pipeline  # 파이프라인 = 일괄 처리
model = make_pipeline(MinMaxScaler(), GridSearchCV(RandomForestClassifier(), parameters, cv=kfold , n_jobs=-1, refit=True, verbose=1))

#3. 컴파일 및 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

#4. 평가 및 예측
from sklearn.metrics import accuracy_score
results = model.score(x_test, y_test)
print("model.score : ", results)  # acc
y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print("accuracy_score : " , round(acc, 2))

print("걸린시간 :", round(end_time - start_time, 2), "초")


'''
#2. 모델 구성 
model = RandomForestClassifier()

#3. 컴파일, 훈련
scores = cross_val_score(model, x_train, y_train, cv = kfold)
print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores,), 4))
'''



# RF
# ACC :  [0.96031746 0.96428571 0.96812749 0.97609562 0.97211155] 
#  평균 ACC :  0.9682

# Grid
# 최적의 매개변수 : RandomForestClassifier(min_samples_split=3, n_jobs=-1)
# 최적의 파라미터 : {'min_samples_split': 3, 'n_jobs': -1}
# best_score : 0.9697780307342061
# model.score : 0.9814814814814815
# accuracy_score :  0.9814814814814815
# 최적 튠 ACC :  0.9814814814814815
# 걸린 시간 : 3.85 초

# Random Search
# 최적의 매개변수 : RandomForestClassifier()
# 최적의 파라미터 : {'min_samples_split': 2}
# best_score : 0.9721589831151585
# model.score : 0.975925925925926
# accuracy_score :  0.975925925925926
# 최적 튠 ACC :  0.975925925925926
# 걸린 시간 : 1.95 초