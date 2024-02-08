# https://dacon.io/competitions/open/235610/mysubmission

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,  KFold, cross_val_score, StratifiedKFold
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#1.데이터
path = "c:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 결측치 처리 
train_csv['type'] = train_csv['type'].map({"white":1, "red":0}).astype(int)
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0}).astype(int)

# x와 y를 분리
x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']

x_train, x_test, y_train, y_test = train_test_split(
x, y,             
train_size=0.7,
random_state=123,
stratify=y,  
shuffle=True,
)
n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123) 
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# scaler = MinMaxScaler() # 클래스 정의
scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성 
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
print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores,), 4))
'''
model.fit(x_train, y_train)


#4.평가, 예측
results = model.score(x_test, y_test) 
y_predict = model.predict(x_test)
 
acc = accuracy_score(y_test, y_predict)
print("acc : ", results)


'''
# LinearSVC                    0.44303030303030305 
# Perceptron                 0.3321212121212121
# LogisticRegression           0.4915151515151515
# KNeighborsClassifier        0.4696969696969697
# DecisionTreeClassifier      0.5793939393939394
# RandomForestClassifier        0.6642424242424242
'''
AdaBoostClassifier 의 정답률 :  0.43
BaggingClassifier 의 정답률 :  0.64
BernoulliNB 의 정답률 :  0.47
CalibratedClassifierCV 의 정답률 :  0.55
DecisionTreeClassifier 의 정답률 :  0.59
DummyClassifier 의 정답률 :  0.44
ExtraTreeClassifier 의 정답률 :  0.57
ExtraTreesClassifier 의 정답률 :  0.66
GaussianNB 의 정답률 :  0.43
GaussianProcessClassifier 의 정답률 :  0.61
GradientBoostingClassifier 의 정답률 :  0.58
HistGradientBoostingClassifier 의 정답률 :  0.65
KNeighborsClassifier 의 정답률 :  0.55
LabelPropagation 의 정답률 :  0.61
LabelSpreading 의 정답률 :  0.61
LinearDiscriminantAnalysis 의 정답률 :  0.56
LinearSVC 의 정답률 :  0.54
LogisticRegression 의 정답률 :  0.55
LogisticRegressionCV 의 정답률 :  0.55
MLPClassifier 의 정답률 :  0.59
NearestCentroid 의 정답률 :  0.26
PassiveAggressiveClassifier 의 정답률 :  0.41
Perceptron 의 정답률 :  0.48
QuadraticDiscriminantAnalysis 의 정답률 :  0.51
RandomForestClassifier 의 정답률 :  0.66
RidgeClassifier 의 정답률 :  0.54
RidgeClassifierCV 의 정답률 :  0.54
SGDClassifier 의 정답률 :  0.51
SVC 의 정답률 :  0.59


'''
'''
KFold
ACC :  [0.55844156 0.52337662 0.55916775 0.52795839 0.58777633] 
 평균 ACC :  0.5513

StratifiedKFold
ACC :  [0.54805195 0.58701299 0.54486346 0.56697009 0.57347204] 
 평균 ACC :  0.5641
'''
