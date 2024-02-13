import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, train_size=0.8, stratify=y)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]}, # 2*3*2 = 12
    {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]}, # 4*4 = 16
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]}, # 4*4 = 16
    {"min_samples_split": [2, 3, 5, 10]}, # 4
    {"n_jobs": [-1,], "min_samples_split": [2, 3, 5, 10]}, # 4 
]   

#2. 모델
# model = SVC()
# model = GridSearchCV(RandomForestClassifier(), 
#                      parameters,
#                      cv=kfold,
#                      verbose=1,
#                      refit=True,
#                     n_jobs=-1,  # CPU
#                      )
model = RandomizedSearchCV(RandomForestClassifier(), 
                     parameters,
                     cv=kfold,
                     verbose=1,
                     refit=True,
                    n_jobs=-1,  # CPU
                    random_state=123,
                    n_iter=20,
                     )
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 :", model.best_estimator_)
# 최적의 매개변수 : SVC(C=1, kernel='linear') 
print("최적의 파라미터 :", model.best_params_)
# 최적의 파라미터 : {'C': 1, 'degree': 3, 'kernel': 'linear'} 우리가 지정한것 중 베스트

print("best_score :", model.best_score_)    # train 스코어
# best_score : 0.975
print("model.score :", model.score(x_test, y_test))
# model.score : 0.9666666666666667

y_predict =model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
                # SVC(C=1, kernel='linear').predict(x_test)
                
print("최적 튠 ACC : ", accuracy_score(y_test, y_pred_best))

print("걸린 시간 :", round(end_time - start_time, 2),"초")

# import pandas as pd
# print(pd.DataFrame(model.cv_results_).T)

# 최적의 매개변수 : RandomForestClassifier(max_depth=30, min_samples_split=10, n_estimators=50)
# 최적의 파라미터 : {'n_estimators': 50, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 30}
# best_score : 0.9583333333333334
# model.score : 0.9666666666666667
# accuracy_score :  0.9666666666666667
# 최적 튠 ACC :  0.9666666666666667
# 걸린 시간 : 1.7 초









