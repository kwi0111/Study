from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    train_size=0.8,
                                                    random_state=123,
                                                    stratify=y,
                                                    shuffle=True
                                                    )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# scaler = MinMaxScaler() # 클래스 정의
scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)   # 정의만 내렸다.
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
#         continue    # 그냥 다음껄로 넘어간다.
#3.컴파일, 훈련
scores = cross_val_score(model, x_train, y_train, cv = kfold)
print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores,), 4)) # ACC :  [0.96666667 0.96666667 1.         0.96666667 0.93333333] 5분할 했으니까 5개 나옴.

y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
print(y_predict) # cv의 예측치
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC :' ,acc)
'''
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("model.score : ", results)  # acc
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : " , acc)

# LinearSVC                     0.9444444444444444
# Perceptron                    1.0
# LogisticRegression            1.0
# KNeighborsClassifier          0.9444444444444444
# DecisionTreeClassifier         0.8888888888888888
# RandomForestClassifier        0.9722222222222222
'''
'''
AdaBoostClassifier 의 정답률 :  0.94
BaggingClassifier 의 정답률 :  0.94
BernoulliNB 의 정답률 :  0.94
CalibratedClassifierCV 의 정답률 :  0.94
DecisionTreeClassifier 의 정답률 :  0.86
DummyClassifier 의 정답률 :  0.39
ExtraTreeClassifier 의 정답률 :  0.94
ExtraTreesClassifier 의 정답률 :  0.97
GaussianNB 의 정답률 :  1.0
GaussianProcessClassifier 의 정답률 :  0.92
GradientBoostingClassifier 의 정답률 :  0.92
HistGradientBoostingClassifier 의 정답률 :  0.97
KNeighborsClassifier 의 정답률 :  0.94
LabelPropagation 의 정답률 :  0.92
LabelSpreading 의 정답률 :  0.92
LinearDiscriminantAnalysis 의 정답률 :  0.97
LinearSVC 의 정답률 :  0.94
LogisticRegression 의 정답률 :  1.0
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  1.0
NearestCentroid 의 정답률 :  0.97
NuSVC 의 정답률 :  1.0
PassiveAggressiveClassifier 의 정답률 :  0.94
Perceptron 의 정답률 :  1.0
QuadraticDiscriminantAnalysis 의 정답률 :  0.97
RandomForestClassifier 의 정답률 :  0.97
RidgeClassifier 의 정답률 :  0.97
RidgeClassifierCV 의 정답률 :  1.0
SGDClassifier 의 정답률 :  0.94
SVC 의 정답률 :  1.0

'''

'''
SVC 5
ACC :  [1.         0.93103448 0.92857143 1.         1.        ] 
 평균 ACC :  0.9719

RandomForestClassifier 5
ACC :  [1.         1.         0.96428571 1.         0.96428571] 
평균 ACC :  0.9857

StratifiedKFold
ACC :  [0.96551724 0.96551724 0.92857143 0.96428571 1.        ] 
 평균 ACC :  0.9648
 
cross_val_predict ACC : 1.0
'''












 




