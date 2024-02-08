# https://dacon.io/competitions/open/236070/overview/description
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
path = "c:\\_data\\dacon\\iris\\"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# x와 y를 분리
x = train_csv.drop(['species'], axis=1)
y = train_csv['species']

x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        train_size=0.7,
        random_state=200,    
        stratify=y,     # 에러 : 분류에서만 쓴다. // y값이 정수로 딱 떨어지는것만 쓴다.
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
y_predict = model.predict(x_test)
 
acc = accuracy_score(y_test, y_predict)
print("acc : ", results)
 
# LinearSVC                     0.9166666666666666
# Perceptron                     0.8333333333333334
# LogisticRegression              1.0
# KNeighborsClassifier            1.0
# DecisionTreeClassifier          1.0
# RandomForestClassifier          1.0

'''   



'''
AdaBoostClassifier 의 정답률 :  1.0
BaggingClassifier 의 정답률 :  1.0
BernoulliNB 의 정답률 :  0.83
CalibratedClassifierCV 의 정답률 :  0.97
DecisionTreeClassifier 의 정답률 :  1.0
DummyClassifier 의 정답률 :  0.33
ExtraTreeClassifier 의 정답률 :  1.0
ExtraTreesClassifier 의 정답률 :  1.0
GaussianNB 의 정답률 :  1.0
GaussianProcessClassifier 의 정답률 :  1.0
GradientBoostingClassifier 의 정답률 :  1.0
HistGradientBoostingClassifier 의 정답률 :  1.0
KNeighborsClassifier 의 정답률 :  1.0
LabelPropagation 의 정답률 :  1.0
LabelSpreading 의 정답률 :  1.0
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  0.97
LogisticRegression 의 정답률 :  1.0
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  1.0
NearestCentroid 의 정답률 :  0.92
NuSVC 의 정답률 :  1.0
PassiveAggressiveClassifier 의 정답률 :  0.86
Perceptron 의 정답률 :  0.97
QuadraticDiscriminantAnalysis 의 정답률 :  1.0
RandomForestClassifier 의 정답률 :  1.0
RidgeClassifier 의 정답률 :  0.81
RidgeClassifierCV 의 정답률 :  0.81
SGDClassifier 의 정답률 :  1.0
SVC 의 정답률 :  1.0


'''

'''
KFold
ACC :  [0.88235294 0.94117647 1.         0.82352941 0.9375    ] 
 평균 ACC :  0.9169
 
StratifiedKFold
ACC :  [0.88235294 0.94117647 0.88235294 1.         0.875     ] 
 평균 ACC :  0.9162
'''


'''
============== AdaBoostClassifier =================
ACC :  [0.94117647 0.88235294 0.88235294 0.94117647 0.8125    ]
 평균 ACC :  0.8919
cross_val_predict ACC : 1.0
============== BaggingClassifier =================
ACC :  [0.82352941 0.94117647 0.88235294 1.         0.75      ]
 평균 ACC :  0.8794
cross_val_predict ACC : 0.9722222222222222
============== BernoulliNB =================
ACC :  [0.64705882 0.70588235 0.82352941 0.70588235 0.75      ]
 평균 ACC :  0.7265
cross_val_predict ACC : 0.8333333333333334
============== CalibratedClassifierCV =================
ACC :  [0.82352941 0.82352941 0.82352941 0.82352941 0.875     ]
 평균 ACC :  0.8338
cross_val_predict ACC : 0.7777777777777778
CategoricalNB 은 안돌아간다!!!
ClassifierChain 은 안돌아간다!!!
ComplementNB 은 안돌아간다!!!
============== DecisionTreeClassifier =================
ACC :  [0.82352941 0.94117647 0.88235294 0.94117647 0.875     ]
 평균 ACC :  0.8926
cross_val_predict ACC : 1.0
============== DummyClassifier =================
ACC :  [0.35294118 0.35294118 0.35294118 0.29411765 0.3125    ]
 평균 ACC :  0.3331
cross_val_predict ACC : 0.2777777777777778
============== ExtraTreeClassifier =================
ACC :  [1.         0.82352941 0.88235294 0.94117647 0.75      ]
 평균 ACC :  0.8794
cross_val_predict ACC : 0.9722222222222222
============== ExtraTreesClassifier =================
ACC :  [0.88235294 0.94117647 0.88235294 1.         0.75      ]
 평균 ACC :  0.8912
cross_val_predict ACC : 1.0
============== GaussianNB =================
ACC :  [0.88235294 0.94117647 0.88235294 1.         0.875     ]
 평균 ACC :  0.9162
cross_val_predict ACC : 1.0
============== GaussianProcessClassifier =================
ACC :  [0.94117647 0.94117647 0.88235294 0.88235294 0.8125    ]
 평균 ACC :  0.8919
cross_val_predict ACC : 0.9722222222222222
============== GradientBoostingClassifier =================
ACC :  [0.82352941 0.88235294 0.88235294 1.         0.75      ]
 평균 ACC :  0.8676
cross_val_predict ACC : 1.0
============== HistGradientBoostingClassifier =================
ACC :  [0.82352941 0.94117647 0.88235294 1.         0.75      ]
 평균 ACC :  0.8794
cross_val_predict ACC : 0.2777777777777778
============== KNeighborsClassifier =================
ACC :  [0.94117647 0.94117647 0.94117647 0.94117647 0.8125    ]
 평균 ACC :  0.9154
cross_val_predict ACC : 0.9444444444444444
============== LabelPropagation =================
ACC :  [0.88235294 0.94117647 0.88235294 0.94117647 0.875     ] 
 평균 ACC :  0.9044
cross_val_predict ACC : 1.0
============== LabelSpreading =================
ACC :  [0.88235294 0.94117647 0.88235294 0.94117647 0.875     ]
 평균 ACC :  0.9044
cross_val_predict ACC : 1.0
============== LinearDiscriminantAnalysis =================
ACC :  [0.94117647 0.94117647 0.88235294 1.         1.        ]
 평균 ACC :  0.9529
cross_val_predict ACC : 1.0
============== LinearSVC =================
ACC :  [1.         0.88235294 0.88235294 0.82352941 0.875     ]
 평균 ACC :  0.8926
cross_val_predict ACC : 0.9444444444444444
============== LogisticRegression =================
ACC :  [0.94117647 0.94117647 0.88235294 0.94117647 0.875     ]
 평균 ACC :  0.9162
cross_val_predict ACC : 0.9444444444444444
============== LogisticRegressionCV =================
ACC :  [0.94117647 0.88235294 0.88235294 0.94117647 0.875     ]
 평균 ACC :  0.9044
cross_val_predict ACC : 1.0
============== MLPClassifier =================
ACC :  [0.88235294 0.88235294 0.88235294 0.88235294 0.875     ]
 평균 ACC :  0.8809
cross_val_predict ACC : 0.9722222222222222
MultiOutputClassifier 은 안돌아간다!!!
MultinomialNB 은 안돌아간다!!!
============== NearestCentroid =================
ACC :  [0.82352941 0.88235294 0.94117647 0.76470588 0.8125    ]
 평균 ACC :  0.8449
cross_val_predict ACC : 0.8611111111111112
============== NuSVC =================
ACC :  [0.88235294 0.94117647 0.88235294 0.94117647 0.875     ]
 평균 ACC :  0.9044
cross_val_predict ACC : 0.9722222222222222
OneVsOneClassifier 은 안돌아간다!!!
OneVsRestClassifier 은 안돌아간다!!!
OutputCodeClassifier 은 안돌아간다!!!
============== PassiveAggressiveClassifier =================
ACC :  [0.64705882 0.88235294 0.88235294 0.76470588 0.625     ]
 평균 ACC :  0.7603
cross_val_predict ACC : 0.9166666666666666
============== Perceptron =================
ACC :  [0.82352941 0.82352941 0.64705882 0.82352941 0.875     ]
 평균 ACC :  0.7985
cross_val_predict ACC : 0.8333333333333334
============== QuadraticDiscriminantAnalysis =================
ACC :  [0.94117647 1.         0.88235294 1.         1.        ]
 평균 ACC :  0.9647
cross_val_predict ACC : 0.9166666666666666
============== RadiusNeighborsClassifier =================
ACC :  [0.88235294 0.94117647 0.94117647        nan 0.8125    ]
 평균 ACC :  nan
cross_val_predict ACC : 0.9722222222222222
============== RandomForestClassifier =================
ACC :  [0.88235294 0.94117647 0.88235294 1.         0.8125    ]
 평균 ACC :  0.9037
cross_val_predict ACC : 1.0
============== RidgeClassifier =================
ACC :  [0.82352941 0.76470588 0.82352941 0.70588235 0.875     ]
 평균 ACC :  0.7985
cross_val_predict ACC : 0.8055555555555556
============== RidgeClassifierCV =================
ACC :  [0.76470588 0.76470588 0.82352941 0.70588235 0.875     ]
 평균 ACC :  0.7868
cross_val_predict ACC : 0.8055555555555556
============== SGDClassifier =================
ACC :  [0.94117647 0.88235294 0.88235294 0.82352941 0.875     ]
 평균 ACC :  0.8809
cross_val_predict ACC : 0.8333333333333334
============== SVC =================
ACC :  [0.88235294 0.94117647 0.88235294 1.         0.875     ]
 평균 ACC :  0.9162
cross_val_predict ACC : 0.9722222222222222
StackingClassifier 은 안돌아간다!!!
VotingClassifier 은 안돌아간다!!!
'''