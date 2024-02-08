from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,cross_val_predict
import pandas as pd
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


#1. 데이터 

#1. 데이터 // 판다스, 넘파이 
path = "C:\\_data\\dacon\\diabetes\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)    
test_csv = pd.read_csv(path + "test.csv", index_col=0)      # 헤더는 기본 첫번째 줄이 디폴트값
submission_csv = pd.read_csv(path + "sample_submission.csv")       

############# x 와 y를 분리 ################
x = train_csv.drop(['Outcome', 'Insulin'], axis=1)   # 행삭제 : axis = 0 // 열삭제 : axis = 1 // train_csv에 있는 'Outcome'열 삭제
y = train_csv.drop(['Insulin'], axis=1)       # 행삭제 : axis = 0 // 열삭제 : axis = 1 // train_csv에 있는 'Outcome'열 삭제 
y = train_csv['Outcome']                      # train_csv에 있는 'Outcome'열을 y로 설정
test_csv = test_csv.drop(['Insulin'], axis=1)



x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.9, stratify=y, random_state=123123,
)
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 3
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
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
#4. 평가, 예측
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("model.score : ", results)  


# LinearSVC                     0.6060606060606061
# Perceptron                  0.7121212121212122
# LogisticRegression          0.7878787878787878
# KNeighborsClassifier        0.7272727272727273
# DecisionTreeClassifier      0.7727272727272727
# RandomForestClassifier        0.803030303030303
'''
'''
daBoostClassifier 의 정답률 :  0.79
BaggingClassifier 의 정답률 :  0.77
BernoulliNB 의 정답률 :  0.68
CalibratedClassifierCV 의 정답률 :  0.8
DecisionTreeClassifier 의 정답률 :  0.82
DummyClassifier 의 정답률 :  0.61
ExtraTreeClassifier 의 정답률 :  0.71
ExtraTreesClassifier 의 정답률 :  0.83
GaussianNB 의 정답률 :  0.76
GaussianProcessClassifier 의 정답률 :  0.8
GradientBoostingClassifier 의 정답률 :  0.77
HistGradientBoostingClassifier 의 정답률 :  0.85
KNeighborsClassifier 의 정답률 :  0.8
LabelPropagation 의 정답률 :  0.71
LabelSpreading 의 정답률 :  0.71
LinearDiscriminantAnalysis 의 정답률 :  0.79
LinearSVC 의 정답률 :  0.8
LogisticRegression 의 정답률 :  0.79
LogisticRegressionCV 의 정답률 :  0.79
MLPClassifier 의 정답률 :  0.83
NearestCentroid 의 정답률 :  0.67
NuSVC 의 정답률 :  0.83
PassiveAggressiveClassifier 의 정답률 :  0.8
Perceptron 의 정답률 :  0.71
QuadraticDiscriminantAnalysis 의 정답률 :  0.74
RandomForestClassifier 의 정답률 :  0.77
RidgeClassifier 의 정답률 :  0.8
RidgeClassifierCV 의 정답률 :  0.8
SGDClassifier 의 정답률 :  0.76
SVC 의 정답률 :  0.82
'''

'''
SVC 3
ACC :  [0.78061224 0.70769231 0.71282051] 
평균 ACC :  0.7337

RandomForestClassifier 3
ACC :  [0.77040816 0.73846154 0.72307692] 
평균 ACC :  0.744

StratifiedKFold
ACC :  [0.73979592 0.72820513 0.77435897] 
 평균 ACC :  0.7475
'''

'''
============== AdaBoostClassifier =================
ACC :  [0.75510204 0.74871795 0.74871795]
 평균 ACC :  0.7508
cross_val_predict ACC : 0.7727272727272727
============== BaggingClassifier =================
ACC :  [0.78061224 0.72820513 0.72307692]
 평균 ACC :  0.744
cross_val_predict ACC : 0.696969696969697
============== BernoulliNB =================
ACC :  [0.73979592 0.68205128 0.70769231]
 평균 ACC :  0.7098
cross_val_predict ACC : 0.6666666666666666
============== CalibratedClassifierCV =================
ACC :  [0.80612245 0.78974359 0.72307692]
 평균 ACC :  0.773
cross_val_predict ACC : 0.7272727272727273
CategoricalNB 은 안돌아간다!!!
ClassifierChain 은 안돌아간다!!!
ComplementNB 은 안돌아간다!!!
============== DecisionTreeClassifier =================
ACC :  [0.72959184 0.71794872 0.66666667]
 평균 ACC :  0.7047
cross_val_predict ACC : 0.6818181818181818
============== DummyClassifier =================
ACC :  [0.64795918 0.67179487 0.63076923]
 평균 ACC :  0.6502
cross_val_predict ACC : 0.6515151515151515
============== ExtraTreeClassifier =================
ACC :  [0.75510204 0.67179487 0.61025641]
 평균 ACC :  0.6791
cross_val_predict ACC : 0.7121212121212122
============== ExtraTreesClassifier =================
ACC :  [0.80612245 0.72820513 0.75384615]
 평균 ACC :  0.7627
cross_val_predict ACC : 0.7727272727272727
============== GaussianNB =================
ACC :  [0.76530612 0.76410256 0.73846154]
 평균 ACC :  0.756
cross_val_predict ACC : 0.7272727272727273
============== GaussianProcessClassifier =================
ACC :  [0.78061224 0.72307692 0.75897436]
 평균 ACC :  0.7542
cross_val_predict ACC : 0.7575757575757576
============== GradientBoostingClassifier =================
ACC :  [0.79081633 0.75897436 0.71282051]
 평균 ACC :  0.7542
cross_val_predict ACC : 0.7272727272727273
============== HistGradientBoostingClassifier =================
ACC :  [0.79081633 0.72820513 0.77435897]
 평균 ACC :  0.7645
cross_val_predict ACC : 0.7575757575757576
============== KNeighborsClassifier =================
ACC :  [0.71938776 0.68717949 0.75897436]
 평균 ACC :  0.7218
cross_val_predict ACC : 0.7878787878787878
============== LabelPropagation =================
ACC :  [0.69387755 0.64615385 0.69230769]
 평균 ACC :  0.6774
cross_val_predict ACC : 0.6666666666666666
============== LabelSpreading =================
ACC :  [0.69387755 0.64615385 0.69230769]
 평균 ACC :  0.6774
cross_val_predict ACC : 0.6666666666666666
============== LinearDiscriminantAnalysis =================
ACC :  [0.80612245 0.78974359 0.73333333]
 평균 ACC :  0.7764
cross_val_predict ACC : 0.7878787878787878
============== LinearSVC =================
ACC :  [0.80612245 0.79487179 0.72820513]
 평균 ACC :  0.7764
cross_val_predict ACC : 0.7878787878787878
============== LogisticRegression =================
ACC :  [0.80102041 0.79487179 0.72307692]
 평균 ACC :  0.773
cross_val_predict ACC : 0.8181818181818182
============== LogisticRegressionCV =================
ACC :  [0.80102041 0.78974359 0.73846154]
 평균 ACC :  0.7764
cross_val_predict ACC : 0.7878787878787878
============== MLPClassifier =================
ACC :  [0.81122449 0.74358974 0.73846154]
 평균 ACC :  0.7644
cross_val_predict ACC : 0.7878787878787878
MultiOutputClassifier 은 안돌아간다!!!
MultinomialNB 은 안돌아간다!!!
============== NearestCentroid =================
ACC :  [0.73979592 0.72307692 0.7025641 ]
 평균 ACC :  0.7218
cross_val_predict ACC : 0.803030303030303
============== NuSVC =================
ACC :  [0.79591837 0.75384615 0.72820513]
 평균 ACC :  0.7593
cross_val_predict ACC : 0.7727272727272727
OneVsOneClassifier 은 안돌아간다!!!
OneVsRestClassifier 은 안돌아간다!!!
OutputCodeClassifier 은 안돌아간다!!!
============== PassiveAggressiveClassifier =================
ACC :  [0.67857143 0.75384615 0.6974359 ]
 평균 ACC :  0.71
cross_val_predict ACC : 0.7424242424242424
============== Perceptron =================
ACC :  [0.68367347 0.67179487 0.71282051]
 평균 ACC :  0.6894
cross_val_predict ACC : 0.7424242424242424
============== QuadraticDiscriminantAnalysis =================
ACC :  [0.75510204 0.74871795 0.71794872]
 평균 ACC :  0.7406
cross_val_predict ACC : 0.7424242424242424
============== RadiusNeighborsClassifier =================
ACC :  [nan nan nan]
 평균 ACC :  nan
RadiusNeighborsClassifier 은 안돌아간다!!!
============== RandomForestClassifier =================
ACC :  [0.79081633 0.75384615 0.73846154]
 평균 ACC :  0.761
cross_val_predict ACC : 0.7727272727272727
============== RidgeClassifier =================
ACC :  [0.81632653 0.78974359 0.72820513]
 평균 ACC :  0.7781
cross_val_predict ACC : 0.803030303030303
============== RidgeClassifierCV =================
ACC :  [0.81632653 0.78974359 0.72820513]
 평균 ACC :  0.7781
cross_val_predict ACC : 0.803030303030303
============== SGDClassifier =================
ACC :  [0.71938776 0.75897436 0.72307692]
 평균 ACC :  0.7338
cross_val_predict ACC : 0.7575757575757576
============== SVC =================
ACC :  [0.81122449 0.75384615 0.73333333]
 평균 ACC :  0.7661
cross_val_predict ACC : 0.803030303030303
StackingClassifier 은 안돌아간다!!!
VotingClassifier 은 안돌아간다!!!
'''