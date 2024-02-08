from sklearn.datasets import load_breast_cancer     # 유방암
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings(action='ignore')

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
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
allAlgorithms = all_estimators(type_filter='classifier')
for name, algorithms in allAlgorithms:
    try:
        #2.모델
        model = algorithms()
        #3.훈련
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        print(name,'의 정답률', acc)
    except:
        continue

'''
#.3 컴파일, 훈련
model.fit(x_train, y_train)

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

