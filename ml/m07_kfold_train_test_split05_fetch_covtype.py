from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    train_size=0.9,
                                                    random_state=123,
                                                    stratify=y,
                                                    shuffle=True
                                                    ) 
n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델구성
# model = RandomForestClassifier()
model = SVC()
# allAlgorithms = all_estimators(type_filter='classifier')

# for name, algorithm in allAlgorithms:
#     try:
#         #2. 모델
#         model = algorithm()
#         #.3 훈련
#         model.fit(x_train, y_train)
        
#         acc = model.score(x_test, y_test)   
#         print(name, "의 정답률 : ", round(acc, 2))   
#     except: 
#         continue
    
#3.컴파일, 훈련
scores = cross_val_score(model, x_train, y_train, cv = kfold)
print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores,), 4)) # ACC :  [0.96666667 0.96666667 1.         0.96666667 0.93333333] 5분할 했으니까 5개 나옴.

y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
print(y_predict) # cv의 예측치
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC :' ,acc)

# model.fit(x_train, y_train)
'''

#4. 평가, 예측
results = model.score(x_test, y_test) 
y_predict = model.predict(x_test)
 
acc = accuracy_score(y_test, y_predict)
print("acc : ", results)

# LinearSVC                     
# Perceptron                  acc :  0.5297580117724002
# LogisticRegression           acc :  0.6179477470655055
# KNeighborsClassifier        acc :  0.9706722660149393
# DecisionTreeClassifier      acc :  0.9428246876183264
# RandomForestClassifier        acc :  0.9572648101614403

'''


'''
RandomForestClassifier 5
ACC :  [0.95363447 0.95288864 0.95233405 0.95445679 0.95274521] 
평균 ACC :  0.9532

StratifiedKFold

'''