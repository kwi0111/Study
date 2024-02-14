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

parameters = [
    {"randomforestclassifier__n_estimators": [100, 200], "randomforestclassifier__max_depth": [6, 10, 12], "randomforestclassifier__min_samples_leaf": [3, 10]}, # 12
    {"randomforestclassifier__max_depth": [6, 8, 10, 12], "randomforestclassifier__min_samples_leaf": [3, 5, 7, 10]}, # 16
    {"randomforestclassifier__min_samples_leaf": [3, 5, 7, 10], "randomforestclassifier__min_samples_split": [2, 3, 5, 10]}, # 16
    {"randomforestclassifier__min_samples_split": [2, 3, 5, 10]},
    {"randomforestclassifier__min_samples_split": [2, 3, 5, 10]}, # 4
]       # 랜포용 파라미터 // 인식시켜주기 위해서 // 잘 안쓴다.

#2. 모델 구성
pipe = make_pipeline(MinMaxScaler(),
                  RandomForestClassifier())  # 하나의 파이프가 모델
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


