from sklearn.datasets import load_breast_cancer     # 유방암
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
                                                    )
n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)   # 정의만 내렸다.
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
# model = RandomForestClassifier()
model = SVC()

# allAlgorithms = all_estimators(type_filter='classifier')
# for name, algorithms in allAlgorithms:
#     try:
#         #2.모델
#         model = algorithms()
#         #3.훈련
#         model.fit(x_train, y_train)
#         acc = model.score(x_test, y_test)
#         print(name,'의 정답률', acc)
#     except:
#         continue

#.3 컴파일, 훈련
scores = cross_val_score(model, x_train, y_train, cv = kfold)
print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores,), 4)) # ACC :  [0.96666667 0.96666667 1.         0.96666667 0.93333333] 5분할 했으니까 5개 나옴.

# model.fit(x_train, y_train)
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

