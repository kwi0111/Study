from sklearn.datasets import load_breast_cancer     # 유방암
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC



#1. 데이터 
datasets = load_breast_cancer()

print(datasets.DESCR)   
print(datasets.feature_names)

x = datasets.data
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    random_state=123,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    stratify=y
                                                    )
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)   # 정의만 내렸다.
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

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

#4. 평가, 예측
results = model.score(x_test, y_test)
print("model.score : ", results)  # acc
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : " , acc)

# LinearSVC                     0.9210526315789473
# Perceptron                  0.8859649122807017
# LogisticRegression           0.9824561403508771
# KNeighborsClassifier        0.9649122807017544
# DecisionTreeClassifier      0.9649122807017544
# RandomForestClassifier        0.9912280701754386
'''
'''
AdaBoostClassifier 의 정답률 0.9736842105263158
BaggingClassifier 의 정답률 0.9824561403508771
BernoulliNB 의 정답률 0.6403508771929824
CalibratedClassifierCV 의 정답률 0.956140350877193
ComplementNB 의 정답률 0.9035087719298246
DecisionTreeClassifier 의 정답률 0.956140350877193
DummyClassifier 의 정답률 0.6403508771929824
ExtraTreeClassifier 의 정답률 0.9473684210526315
ExtraTreesClassifier 의 정답률 0.9912280701754386
GaussianNB 의 정답률 0.956140350877193
GaussianProcessClassifier 의 정답률 0.9298245614035088
GradientBoostingClassifier 의 정답률 0.9736842105263158
HistGradientBoostingClassifier 의 정답률 0.9736842105263158
KNeighborsClassifier 의 정답률 0.9649122807017544
LabelPropagation 의 정답률 0.38596491228070173
LabelSpreading 의 정답률 0.38596491228070173
LinearDiscriminantAnalysis 의 정답률 0.9736842105263158
LinearSVC 의 정답률 0.956140350877193
LogisticRegression 의 정답률 0.9824561403508771
LogisticRegressionCV 의 정답률 0.9912280701754386
MLPClassifier 의 정답률 0.9473684210526315
MultinomialNB 의 정답률 0.9035087719298246
NearestCentroid 의 정답률 0.8771929824561403
NuSVC 의 정답률 0.8596491228070176
PassiveAggressiveClassifier 의 정답률 0.9035087719298246
Perceptron 의 정답률 0.8859649122807017
QuadraticDiscriminantAnalysis 의 정답률 0.9912280701754386
RandomForestClassifier 의 정답률 0.9912280701754386
RidgeClassifier 의 정답률 0.9649122807017544
RidgeClassifierCV 의 정답률 0.9649122807017544
SGDClassifier 의 정답률 0.9298245614035088
SVC 의 정답률 0.9298245614035088

'''

'''
SVC 15
ACC :  [0.96774194 1.         1.         1.         0.96774194 0.93333333
 1.         1.         0.93333333 0.93333333 0.93333333 0.96666667
 0.96666667 1.         0.96666667]
 평균 ACC :  0.9713
 
RandomForestClassifier 5
 ACC :  [0.94505495 0.98901099 0.93406593 0.96703297 0.91208791] 
 평균 ACC :  0.9495
 
 StratifiedKFold
 ACC :  [0.95604396 0.95604396 0.97802198 1.         0.94505495]
 평균 ACC :  0.967
'''

'''
============== AdaBoostClassifier =================
ACC :  [0.96703297 0.97802198 0.97802198 0.96703297 0.97802198]
 평균 ACC :  0.9736
cross_val_predict ACC : 0.9649122807017544
============== BaggingClassifier =================
ACC :  [0.93406593 0.97802198 0.93406593 0.93406593 0.91208791]
 평균 ACC :  0.9385
cross_val_predict ACC : 0.9385964912280702
============== BernoulliNB =================
ACC :  [0.93406593 0.95604396 0.9010989  0.93406593 0.91208791]
 평균 ACC :  0.9275
cross_val_predict ACC : 0.9473684210526315
============== CalibratedClassifierCV =================
ACC :  [0.95604396 0.95604396 0.95604396 0.96703297 0.96703297]
 평균 ACC :  0.9604
cross_val_predict ACC : 0.9473684210526315
CategoricalNB 은 안돌아간다!!!
ClassifierChain 은 안돌아간다!!!
ComplementNB 은 안돌아간다!!!
============== DecisionTreeClassifier =================
ACC :  [0.92307692 0.92307692 0.93406593 0.94505495 0.91208791]
 평균 ACC :  0.9275
cross_val_predict ACC : 0.9385964912280702
============== DummyClassifier =================
ACC :  [0.58241758 0.59340659 0.61538462 0.68131868 0.64835165]
 평균 ACC :  0.6242
cross_val_predict ACC : 0.6403508771929824
============== ExtraTreeClassifier =================
ACC :  [0.86813187 0.96703297 0.9010989  0.92307692 0.89010989]
 평균 ACC :  0.9099
cross_val_predict ACC : 0.956140350877193
============== ExtraTreesClassifier =================
ACC :  [0.95604396 0.97802198 0.95604396 0.95604396 0.94505495]
 평균 ACC :  0.9582
cross_val_predict ACC : 0.9824561403508771
============== GaussianNB =================
ACC :  [0.91208791 0.98901099 0.87912088 0.95604396 0.9010989 ]
 평균 ACC :  0.9275
cross_val_predict ACC : 0.956140350877193
============== GaussianProcessClassifier =================
ACC :  [0.96703297 0.96703297 0.92307692 0.96703297 0.93406593]
 평균 ACC :  0.9516
cross_val_predict ACC : 0.9473684210526315
============== GradientBoostingClassifier =================
ACC :  [0.92307692 0.96703297 0.95604396 0.97802198 0.93406593]
 평균 ACC :  0.9516
cross_val_predict ACC : 0.9385964912280702
============== HistGradientBoostingClassifier =================
ACC :  [0.97802198 0.97802198 0.96703297 0.97802198 0.94505495]
 평균 ACC :  0.9692
cross_val_predict ACC : 0.9385964912280702
============== KNeighborsClassifier =================
ACC :  [0.96703297 0.97802198 0.93406593 0.96703297 0.94505495]
 평균 ACC :  0.9582
cross_val_predict ACC : 0.956140350877193
============== LabelPropagation =================
ACC :  [0.96703297 0.95604396 0.93406593 0.95604396 0.93406593]
 평균 ACC :  0.9495
cross_val_predict ACC : 0.9473684210526315
============== LabelSpreading =================
ACC :  [0.96703297 0.95604396 0.93406593 0.95604396 0.93406593]
 평균 ACC :  0.9495
cross_val_predict ACC : 0.9473684210526315
============== LinearDiscriminantAnalysis =================
ACC :  [0.96703297 0.94505495 0.92307692 0.93406593 0.95604396]
 평균 ACC :  0.9451
cross_val_predict ACC : 0.8859649122807017
============== LinearSVC =================
ACC :  [1.         0.97802198 0.97802198 0.96703297 0.94505495]
 평균 ACC :  0.9736
cross_val_predict ACC : 0.956140350877193
============== LogisticRegression =================
ACC :  [1.         0.97802198 0.98901099 0.96703297 0.95604396]
 평균 ACC :  0.978
cross_val_predict ACC : 0.9736842105263158
============== LogisticRegressionCV =================
ACC :  [1.         0.97802198 0.97802198 0.95604396 0.95604396]
 평균 ACC :  0.9736
cross_val_predict ACC : 0.9649122807017544
============== MLPClassifier =================
ACC :  [1.         0.96703297 1.         0.95604396 0.95604396]
 평균 ACC :  0.9758
cross_val_predict ACC : 0.9824561403508771
MultiOutputClassifier 은 안돌아간다!!!
MultinomialNB 은 안돌아간다!!!
============== NearestCentroid =================
ACC :  [0.93406593 0.96703297 0.85714286 0.94505495 0.91208791]
 평균 ACC :  0.9231
cross_val_predict ACC : 0.956140350877193
============== NuSVC =================
ACC :  [0.96703297 0.96703297 0.87912088 0.95604396 0.91208791]
 평균 ACC :  0.9363
cross_val_predict ACC : 0.956140350877193
OneVsOneClassifier 은 안돌아간다!!!
OneVsRestClassifier 은 안돌아간다!!!
OutputCodeClassifier 은 안돌아간다!!!
============== PassiveAggressiveClassifier =================
ACC :  [0.96703297 0.95604396 0.95604396 0.95604396 0.94505495]
 평균 ACC :  0.956
cross_val_predict ACC : 0.956140350877193
============== Perceptron =================
ACC :  [0.95604396 0.97802198 0.95604396 0.95604396 0.95604396]
 평균 ACC :  0.9604
cross_val_predict ACC : 0.9473684210526315
============== QuadraticDiscriminantAnalysis =================
ACC :  [0.95604396 0.97802198 0.92307692 0.93406593 0.96703297]
 평균 ACC :  0.9516
cross_val_predict ACC : 0.8508771929824561
============== RadiusNeighborsClassifier =================
ACC :  [nan nan nan nan nan]
 평균 ACC :  nan
RadiusNeighborsClassifier 은 안돌아간다!!!
============== RandomForestClassifier =================
ACC :  [0.93406593 0.97802198 0.95604396 0.96703297 0.92307692]
 평균 ACC :  0.9516
cross_val_predict ACC : 0.9385964912280702
============== RidgeClassifier =================
ACC :  [0.96703297 0.95604396 0.93406593 0.95604396 0.95604396]
 평균 ACC :  0.9538
cross_val_predict ACC : 0.9385964912280702
============== RidgeClassifierCV =================
ACC :  [0.96703297 0.94505495 0.93406593 0.95604396 0.93406593]
 평균 ACC :  0.9473
cross_val_predict ACC : 0.956140350877193
============== SGDClassifier =================
ACC :  [0.98901099 0.97802198 0.96703297 0.95604396 0.94505495]
 평균 ACC :  0.967
cross_val_predict ACC : 0.9473684210526315
============== SVC =================
ACC :  [0.98901099 0.97802198 0.94505495 0.95604396 0.97802198]
 평균 ACC :  0.9692
cross_val_predict ACC : 0.9649122807017544
StackingClassifier 은 안돌아간다!!!
VotingClassifier 은 안돌아간다!!!
'''


