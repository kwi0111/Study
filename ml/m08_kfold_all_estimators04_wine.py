import numpy as np
from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings(action='ignore')

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
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)   # 정의만 내렸다.
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


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
'''
'''
============== AdaBoostClassifier =================
ACC :  [0.89655172 0.86206897 0.85714286 0.96428571 0.89285714]
 평균 ACC :  0.8946
cross_val_predict ACC : 0.8333333333333334
============== BaggingClassifier =================
ACC :  [0.96551724 0.89655172 0.89285714 0.96428571 0.92857143]
 평균 ACC :  0.9296
cross_val_predict ACC : 0.9444444444444444
============== BernoulliNB =================
ACC :  [0.96551724 0.89655172 0.85714286 0.96428571 0.92857143]
 평균 ACC :  0.9224
cross_val_predict ACC : 0.9444444444444444
============== CalibratedClassifierCV =================
ACC :  [0.96551724 0.96551724 1.         1.         1.        ]
 평균 ACC :  0.9862
cross_val_predict ACC : 0.9444444444444444
CategoricalNB 은 안돌아간다!!!
ClassifierChain 은 안돌아간다!!!
ComplementNB 은 안돌아간다!!!
============== DecisionTreeClassifier =================
ACC :  [0.93103448 0.86206897 0.92857143 0.89285714 0.89285714]
 평균 ACC :  0.9015
cross_val_predict ACC : 0.8333333333333334
============== DummyClassifier =================
ACC :  [0.37931034 0.4137931  0.42857143 0.35714286 0.42857143]
 평균 ACC :  0.4015
cross_val_predict ACC : 0.2777777777777778
============== ExtraTreeClassifier =================
ACC :  [0.75862069 0.82758621 0.92857143 1.         0.78571429]
 평균 ACC :  0.8601
cross_val_predict ACC : 0.8055555555555556
============== ExtraTreesClassifier =================
ACC :  [1.         1.         1.         1.         0.92857143]
 평균 ACC :  0.9857
cross_val_predict ACC : 0.9166666666666666
============== GaussianNB =================
ACC :  [1.         0.96551724 0.92857143 0.96428571 0.96428571]
 평균 ACC :  0.9645
cross_val_predict ACC : 0.9722222222222222
============== GaussianProcessClassifier =================
ACC :  [1.         1.         0.92857143 0.96428571 0.92857143]
 평균 ACC :  0.9643
cross_val_predict ACC : 0.9444444444444444
============== GradientBoostingClassifier =================
ACC :  [1.         0.86206897 0.92857143 1.         0.92857143]
 평균 ACC :  0.9438
cross_val_predict ACC : 0.8888888888888888
============== HistGradientBoostingClassifier =================
ACC :  [0.96551724 0.96551724 0.96428571 1.         0.92857143]
 평균 ACC :  0.9648
cross_val_predict ACC : 0.2777777777777778
============== KNeighborsClassifier =================
ACC :  [1.         0.93103448 0.92857143 1.         0.92857143]
 평균 ACC :  0.9576
cross_val_predict ACC : 0.9444444444444444
============== LabelPropagation =================
ACC :  [1.         0.96551724 0.92857143 0.96428571 0.89285714]
 평균 ACC :  0.9502
cross_val_predict ACC : 0.8888888888888888
============== LabelSpreading =================
ACC :  [1.         0.96551724 0.92857143 0.96428571 0.89285714]
 평균 ACC :  0.9502
cross_val_predict ACC : 0.8888888888888888
============== LinearDiscriminantAnalysis =================
ACC :  [1.         1.         1.         0.96428571 1.        ]
 평균 ACC :  0.9929
cross_val_predict ACC : 0.8888888888888888
============== LinearSVC =================
ACC :  [0.96551724 0.96551724 1.         1.         1.        ]
 평균 ACC :  0.9862
cross_val_predict ACC : 0.9444444444444444
============== LogisticRegression =================
ACC :  [0.96551724 0.96551724 0.96428571 0.96428571 1.        ]
 평균 ACC :  0.9719
cross_val_predict ACC : 0.9444444444444444
============== LogisticRegressionCV =================
ACC :  [0.96551724 0.89655172 0.92857143 0.96428571 1.        ]
 평균 ACC :  0.951
cross_val_predict ACC : 0.9722222222222222
============== MLPClassifier =================
ACC :  [0.96551724 0.96551724 0.92857143 0.96428571 0.96428571]
 평균 ACC :  0.9576
cross_val_predict ACC : 0.9166666666666666
MultiOutputClassifier 은 안돌아간다!!!
MultinomialNB 은 안돌아간다!!!
============== NearestCentroid =================
ACC :  [1.         0.96551724 0.92857143 0.96428571 0.92857143]
 평균 ACC :  0.9574
cross_val_predict ACC : 0.9722222222222222
============== NuSVC =================
ACC :  [1.         0.93103448 0.89285714 1.         0.96428571]
 평균 ACC :  0.9576
cross_val_predict ACC : 1.0
OneVsOneClassifier 은 안돌아간다!!!
OneVsRestClassifier 은 안돌아간다!!!
OutputCodeClassifier 은 안돌아간다!!!
============== PassiveAggressiveClassifier =================
ACC :  [0.96551724 0.96551724 1.         1.         1.        ]
 평균 ACC :  0.9862
cross_val_predict ACC : 0.9444444444444444
============== Perceptron =================
ACC :  [1.         0.96551724 0.96428571 1.         1.        ]
 평균 ACC :  0.986
cross_val_predict ACC : 1.0
============== QuadraticDiscriminantAnalysis =================
ACC :  [1.         1.         1.         0.96428571 1.        ]
 평균 ACC :  0.9929
cross_val_predict ACC : 0.6111111111111112
============== RadiusNeighborsClassifier =================
ACC :  [nan nan nan nan nan]
 평균 ACC :  nan
RadiusNeighborsClassifier 은 안돌아간다!!!
============== RandomForestClassifier =================
ACC :  [1.         1.         0.96428571 1.         0.96428571]
 평균 ACC :  0.9857
cross_val_predict ACC : 0.9444444444444444
============== RidgeClassifier =================
ACC :  [0.96551724 1.         1.         1.         1.        ]
 평균 ACC :  0.9931
cross_val_predict ACC : 0.9166666666666666
============== RidgeClassifierCV =================
ACC :  [0.96551724 1.         1.         1.         1.        ]
 평균 ACC :  0.9931
cross_val_predict ACC : 0.9166666666666666
============== SGDClassifier =================
ACC :  [0.96551724 0.96551724 1.         1.         1.        ]
 평균 ACC :  0.9862
cross_val_predict ACC : 0.9166666666666666
============== SVC =================
ACC :  [1.         0.93103448 0.92857143 1.         1.        ]
 평균 ACC :  0.9719
cross_val_predict ACC : 1.0
StackingClassifier 은 안돌아간다!!!
VotingClassifier 은 안돌아간다!!!
'''











 




