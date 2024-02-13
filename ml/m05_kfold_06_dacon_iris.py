# https://dacon.io/competitions/open/236070/overview/description
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


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


# .3 컴파일, 훈련
scores = cross_val_score(model, x_train, y_train, cv = kfold)
print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores,), 4))
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