########## 그리드서치, 랜덤 서치, 하빙그리드 적용해서 13개 #############

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline  # 파이프라인 = 일괄 처리 / 함수
from sklearn.pipeline import Pipeline  # 클래스 /
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
import time
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV


#1. 데이터 
x,y = load_iris(return_X_y=True) 

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    random_state=123,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    )

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
# parameters = [
#     {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]}, # 12
#     {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]}, # 16
#     {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]}, # 16
#     {"min_samples_split": [2, 3, 5, 10]},
#     {"n_jobs": [-1,], "min_samples_split": [2, 3, 5, 10]}, 
# ]    
# parameters = {
#     "n_estimators": [10, 50, 100, 200],
#     "max_depth": [None, 10, 20, 30],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4]
# }
parameters = [
    {"RF__n_estimators": [100, 200], "RF__max_depth": [6, 10, 12], "RF__min_samples_leaf": [3, 10]}, # 12
    {"RF__max_depth": [6, 8, 10, 12], "RF__min_samples_leaf": [3, 5, 7, 10]}, # 16
    {"RF__min_samples_leaf": [3, 5, 7, 10], "RF__min_samples_split": [2, 3, 5, 10]}, # 16
    {"RF__min_samples_split": [2, 3, 5, 10]},
    {"RF__min_samples_split": [2, 3, 5, 10]}, # 4
]    

#2. 모델 구성
# model = RandomForestClassifier()
# model = make_pipeline(MinMaxScaler(), RandomForestClassifier())     # 알아서 합쳐진다 (스케일러 필요 x) // 스케일러 기준 (?)
pipe = Pipeline([('MinMax', MinMaxScaler()),
                  ('RF', RandomForestClassifier())])    # 하나의 파이프가 모델
# model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
# model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1, factor=5)


#.3 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


#4. 평가, 예측
results = model.score(x_test, y_test)
print("model.score : ", results)  # acc
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : " , round(acc, 2))


