########## 그리드서치, 랜덤 서치, 하빙그리드 적용해서 13개 #############

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline  # 파이프라인 = 일괄 처리
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV
import time


#1. 데이터 
x,y = load_iris(return_X_y=True) 

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    random_state=123,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    )
print(np.min(x_train), np.max(x_train)) # 0.1 7.9
print(np.min(x_test), np.max(x_test))   # 0.1 7.7


n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
# parameters = [
#     {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]}, # 12
#     {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]}, # 16
#     {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]}, # 16
#     {"min_samples_split": [2, 3, 5, 10]},
#     {"n_jobs": [-1,], "min_samples_split": [2, 3, 5, 10]}, 
# ]    
parameters = {
    "n_estimators": [10, 50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

#2. 모델 구성
# model = RandomForestClassifier()
model = make_pipeline(MinMaxScaler(), GridSearchCV(RandomForestClassifier(), parameters, cv=kfold , n_jobs=-1, refit=True, verbose=1))
# 알아서 합쳐진다 (스케일러 필요 x) // 스케일러 기준 (?)

#.3 컴파일, 훈련
# RF = RandomForestClassifier()
# model = GridSearchCV(RF, param_grid=parameters, cv=kfold , n_jobs=-1, refit=True, verbose=1)
# model = RandomizedSearchCV(RF, parameters, cv=kfold , n_jobs=-1, refit=True, verbose=1)
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


# print(np.min(x_train), np.max(x_train)) # 0.1 7.9
# print(np.min(x_test), np.max(x_test))   # 0.1 7.7     시스템상에서만 바뀐다.



# LinearSVC                     0.9210526315789473
# Perceptron                  0.8859649122807017
# LogisticRegression           0.9824561403508771
# KNeighborsClassifier        0.9649122807017544
# DecisionTreeClassifier      0.9649122807017544
# RandomForestClassifier        0.9912280701754386


'''
for문
LinearSVC score :  0.97
LinearSVC predict :  0.96
Perceptron score :  0.87
Perceptron predict :  0.83
LogisticRegression score :  1.0
LogisticRegression predict :  1.0
KNeighborsClassifier score :  0.97
KNeighborsClassifier predict :  0.96
DecisionTreeClassifier score :  0.93
DecisionTreeClassifier predict :  0.92
RandomForestClassifier score :  0.93
RandomForestClassifier predict :  0.92
'''
