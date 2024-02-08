# https://dacon.io/competitions/open/235610/mysubmission

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings(action='ignore')

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
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123) 
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

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
allAlgorithms = all_estimators(type_filter='classifier')    # SVC 분류형 모델
# allAlgorithms = all_estimators(type_filter='regressor')   # SVR 회귀형(예측) 모델

print("allAlgorithms: ", allAlgorithms)     # 리스트 1개, 튜플 41개(모델 이름1, 클래스1)
print("모델 갯수: ", len(allAlgorithms))    # 분류 모델 갯수:  41

# Iterator만 for문 사용 가능 //  순서대로 다음 값을 리턴할 수 있는 객체
for name, algorithm in allAlgorithms:
    try:
        #2. 모델
        model = algorithm()
        #.3 훈련
        scores = cross_val_score(model, x_train, y_train, cv = kfold)
        print("==============", name, "=================")
        print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores,), 4)) # ACC :  [0.96666667 0.96666667 1.         0.96666667 0.93333333] 5분할 했으니까 5개 나옴.

        y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

        acc = accuracy_score(y_test, y_predict)
        print('cross_val_predict ACC :' ,acc)
    except:
        print(name, '은 안돌아간다!!!')  
        # continue    #그냥 다음껄로 넘어간다.
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


'''
============== AdaBoostClassifier =================
ACC :  [0.45194805 0.47532468 0.41482445 0.46684005 0.34980494]
 평균 ACC :  0.4317
cross_val_predict ACC : 0.5012121212121212
============== BaggingClassifier =================
ACC :  [0.60649351 0.58051948 0.59947984 0.61508453 0.62678804]
 평균 ACC :  0.6057
cross_val_predict ACC : 0.536969696969697
============== BernoulliNB =================
ACC :  [0.45714286 0.46233766 0.45123537 0.45123537 0.45123537]
 평균 ACC :  0.4546
cross_val_predict ACC : 0.45515151515151514
============== CalibratedClassifierCV =================
ACC :  [0.54415584 0.5        0.54226268 0.526658   0.56046814]
 평균 ACC :  0.5347
cross_val_predict ACC : 0.5363636363636364
CategoricalNB 은 안돌아간다!!!
ClassifierChain 은 안돌아간다!!!
ComplementNB 은 안돌아간다!!!
============== DecisionTreeClassifier =================
ACC :  [0.54415584 0.54025974 0.5396619  0.51235371 0.55006502]
 평균 ACC :  0.5373
cross_val_predict ACC : 0.4727272727272727
============== DummyClassifier =================
ACC :  [0.44935065 0.43636364 0.43823147 0.42262679 0.45123537]
 평균 ACC :  0.4396
cross_val_predict ACC : 0.4393939393939394
============== ExtraTreeClassifier =================
ACC :  [0.55194805 0.51298701 0.53055917 0.53576073 0.55136541]
 평균 ACC :  0.5365
cross_val_predict ACC : 0.46545454545454545
============== ExtraTreesClassifier =================
ACC :  [0.66103896 0.61428571 0.64759428 0.62548765 0.66840052]
 평균 ACC :  0.6434
cross_val_predict ACC : 0.5842424242424242
============== GaussianNB =================
ACC :  [0.43766234 0.37142857 0.42912874 0.43693108 0.41612484]
 평균 ACC :  0.4183
cross_val_predict ACC : 0.37272727272727274
============== GaussianProcessClassifier =================
ACC :  [0.59350649 0.57532468 0.5786736  0.58647594 0.58647594]
 평균 ACC :  0.5841
cross_val_predict ACC : 0.5575757575757576
============== GradientBoostingClassifier =================
ACC :  [0.56883117 0.54155844 0.56176853 0.55526658 0.61248375]
 평균 ACC :  0.568
cross_val_predict ACC : 0.5503030303030303
============== HistGradientBoostingClassifier =================
ACC :  [0.6038961  0.6012987  0.60598179 0.6046814  0.65149545]
 평균 ACC :  0.6135
cross_val_predict ACC : 0.5624242424242424
============== KNeighborsClassifier =================
ACC :  [0.52207792 0.50519481 0.51885566 0.51495449 0.54226268]
 평균 ACC :  0.5207
cross_val_predict ACC : 0.516969696969697
============== LabelPropagation =================
ACC :  [0.61168831 0.54545455 0.56827048 0.55656697 0.58907672]
 평균 ACC :  0.5742
cross_val_predict ACC : 0.48424242424242425
============== LabelSpreading =================
ACC :  [0.61298701 0.54675325 0.57347204 0.55786736 0.58907672]
 평균 ACC :  0.576
cross_val_predict ACC : 0.48545454545454547
============== LinearDiscriminantAnalysis =================
ACC :  [0.54415584 0.5012987  0.53185956 0.4993498  0.56046814]
 평균 ACC :  0.5274
cross_val_predict ACC : 0.5515151515151515
============== LinearSVC =================
ACC :  [0.53506494 0.4961039  0.53576073 0.50585176 0.54876463]
 평균 ACC :  0.5243
cross_val_predict ACC : 0.5290909090909091
============== LogisticRegression =================
ACC :  [0.54805195 0.50909091 0.53576073 0.51625488 0.56827048]
 평균 ACC :  0.5355
cross_val_predict ACC : 0.5478787878787879
============== LogisticRegressionCV =================
ACC :  [0.54285714 0.50909091 0.5396619  0.51885566 0.56827048]
 평균 ACC :  0.5357
LogisticRegressionCV 은 안돌아간다!!!
============== MLPClassifier =================
ACC :  [0.56623377 0.51948052 0.54356307 0.54746424 0.57607282]
 평균 ACC :  0.5506
cross_val_predict ACC : 0.5684848484848485
MultiOutputClassifier 은 안돌아간다!!!
MultinomialNB 은 안돌아간다!!!
============== NearestCentroid =================
ACC :  [0.25714286 0.26233766 0.2106632  0.25617685 0.25227568]
 평균 ACC :  0.2477
cross_val_predict ACC : 0.23939393939393938
NuSVC 은 안돌아간다!!!
OneVsOneClassifier 은 안돌아간다!!!
OneVsRestClassifier 은 안돌아간다!!!
OutputCodeClassifier 은 안돌아간다!!!
============== PassiveAggressiveClassifier =================
ACC :  [0.49480519 0.41948052 0.39011704 0.36410923 0.32119636]
 평균 ACC :  0.3979
cross_val_predict ACC : 0.3933333333333333
============== Perceptron =================
ACC :  [0.37532468 0.4        0.40962289 0.43042913 0.39401821]
 평균 ACC :  0.4019
cross_val_predict ACC : 0.46060606060606063
============== QuadraticDiscriminantAnalysis =================
ACC :  [0.45194805 0.44675325 0.47204161 0.46944083 0.46553966]
 평균 ACC :  0.4611
QuadraticDiscriminantAnalysis 은 안돌아간다!!!
============== RadiusNeighborsClassifier =================
ACC :  [nan nan nan nan nan]
 평균 ACC :  nan
RadiusNeighborsClassifier 은 안돌아간다!!!
============== RandomForestClassifier =================
ACC :  [0.65194805 0.62077922 0.63849155 0.64109233 0.65929779]
 평균 ACC :  0.6423
cross_val_predict ACC : 0.5806060606060606
============== RidgeClassifier =================
ACC :  [0.53636364 0.4961039  0.53576073 0.50715215 0.54876463]
 평균 ACC :  0.5248
cross_val_predict ACC : 0.5333333333333333
============== RidgeClassifierCV =================
ACC :  [0.53506494 0.4961039  0.53576073 0.50845254 0.55006502]
 평균 ACC :  0.5251
cross_val_predict ACC : 0.5284848484848484
============== SGDClassifier =================
ACC :  [0.48441558 0.46883117 0.50845254 0.42912874 0.48374512]
 평균 ACC :  0.4749
cross_val_predict ACC : 0.4781818181818182
============== SVC =================
ACC :  [0.55844156 0.52337662 0.55916775 0.52795839 0.58777633]
 평균 ACC :  0.5513
cross_val_predict ACC : 0.5557575757575758
StackingClassifier 은 안돌아간다!!!
VotingClassifier 은 안돌아간다!!!
'''