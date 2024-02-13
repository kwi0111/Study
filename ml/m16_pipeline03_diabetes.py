#https://dacon.io/competitions/open/236068/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC #softvector machine
from sklearn.linear_model import Perceptron, LogisticRegression , LinearRegression#분류!
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import make_pipeline  # 파이프라인 = 일괄 처리


#1. 데이터
path = "c:\\_data\\dacon\\diabetes\\"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)
test_csv = pd.read_csv(path +"test.csv",index_col=0  ).drop(['Pregnancies', 'DiabetesPedigreeFunction'], axis=1)

test = train_csv['SkinThickness']
for i in range(test.size):
      if test[i] == 0:
         test[i] =test.mean()
        
#train_csv['BloodPressure'] = test


submission_csv = pd.read_csv(path + "sample_submission.csv") 


print(train_csv.columns) #'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
      # 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
      

  

x = train_csv.drop(['Outcome','Pregnancies','DiabetesPedigreeFunction'], axis=1)

#print(x)
y = train_csv['Outcome']
#print(y)

print(np.unique(y, return_counts= True)) #(array([0, 1]), array([424, 228])
print(pd.Series(y).value_counts()) #다 똑가틈
print(pd.DataFrame(y).value_counts()) #다 똑가틈 (dataframe을 쓰면 메모리를 더 씀,행렬데이터는 dataframe만 가능)
print(pd.value_counts(y))

from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
#n_split = 섞어서 분할하는 갯수

#2. 모델구성
parameters = [
    {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1, 2, 4], "min_samples_split": [2, 3, 5, 10]},
]    

model = make_pipeline(MinMaxScaler(), GridSearchCV(RandomForestClassifier(), parameters, cv=kfold , n_jobs=-1, refit=True, verbose=1))
# model = HalvingGridSearchCV(RandomForestClassifier(), 
#                      parameters,
#                      cv=kfold,
#                      verbose=1,
#                      refit=True,
#                      n_jobs=-1, 
#                      random_state=66,
#                      # n_iter=20,  # 디폴트 10
#                      factor=4, # 디폴트 3 
#                      min_resources=30,  # 데이터 조절하고싶으면 factor, min_resources 맞춘다.
#                      )

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

# 최적의 매개변수 :  RandomForestClassifier(max_depth=12, min_samples_leaf=10)
# 최적의 파라미터 :  {'max_depth': 12, 'min_samples_leaf': 10, 'n_estimators': 100}
# best_score : 0.7792490842490842
# score : 0.732824427480916
# accuracy_score : 0.732824427480916
# 최적튠 ACC : 0.732824427480916
# 걸린시간 : 8.23 초


# 최적의 매개변수 :  RandomForestClassifier(max_depth=12, min_samples_leaf=7)
# 최적의 파라미터 :  {'min_samples_leaf': 7, 'max_depth': 12}
# best_score : 0.7792673992673993
# score : 0.7251908396946565
# accuracy_score : 0.7251908396946565
# 최적튠 ACC : 0.7251908396946565
# 걸린시간 : 2.57 초

# n_iterations: 3
# n_required_iterations: 3
# n_possible_iterations: 3
# min_resources_: 30
# max_resources_: 521
# aggressive_elimination: False
# factor: 4
# ----------
# iter: 0
# n_candidates: 60
# n_resources: 30
# Fitting 5 folds for each of 60 candidates, totalling 300 fits
# ----------
# iter: 1
# n_candidates: 15
# n_resources: 120
# Fitting 5 folds for each of 15 candidates, totalling 75 fits
# ----------
# iter: 2
# n_candidates: 4
# n_resources: 480
# Fitting 5 folds for each of 4 candidates, totalling 20 fits
# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_leaf=7)
# 최적의 파라미터 :  {'max_depth': 6, 'min_samples_leaf': 7}
# best_score : 0.7773245614035087
# score : 0.7251908396946565
# accuracy_score : 0.7251908396946565
# 최적튠 ACC : 0.7251908396946565
# 걸린시간 : 3.95 초