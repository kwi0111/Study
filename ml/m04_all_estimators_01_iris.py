from sklearn.datasets import load_iris
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape) # (150, 4) (150,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,           
    train_size=0.8,
    random_state=1234,     
    stratify=y,   
    shuffle=True,
    )


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
        model.fit(x_train, y_train)
        
        acc = model.score(x_test, y_test)   
        print(name, "의 정답률 : ", acc)   
    except:
        # print(name, '은 바보 멍충이!!!')  
        continue    #그냥 다음껄로 넘어간다.
              
'''
#.3 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
# print("model.score : ", results)  # acc
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : " , acc)
'''


# LinearSVC
# accuracy_score :  0.8666666666666667

# Perceptron
# accuracy_score :  0.7666666666666667

# LogisticRegression
# accuracy_score :  0.9333333333333333

# KNeighborsClassifier
# accuracy_score :  0.9333333333333333

# DecisionTreeClassifier
# accuracy_score :  0.9333333333333333

# RandomForestClassifier
# accuracy_score :  0.9333333333333333
'''
AdaBoostClassifier 의 정답률 :  0.9333333333333333
BaggingClassifier 의 정답률 :  0.9333333333333333
BernoulliNB 의 정답률 :  0.3333333333333333
CalibratedClassifierCV 의 정답률 :  0.8666666666666667
CategoricalNB 의 정답률 :  0.9666666666666667
ComplementNB 의 정답률 :  0.6666666666666666
DecisionTreeClassifier 의 정답률 :  0.9333333333333333
DummyClassifier 의 정답률 :  0.3333333333333333
ExtraTreeClassifier 의 정답률 :  0.9333333333333333
ExtraTreesClassifier 의 정답률 :  0.9
GaussianNB 의 정답률 :  0.9
GaussianProcessClassifier 의 정답률 :  0.9666666666666667
GradientBoostingClassifier 의 정답률 :  0.9333333333333333
HistGradientBoostingClassifier 의 정답률 :  0.9333333333333333
KNeighborsClassifier 의 정답률 :  0.9333333333333333
LabelPropagation 의 정답률 :  0.9
LabelSpreading 의 정답률 :  0.9
LinearDiscriminantAnalysis 의 정답률 :  0.9666666666666667
LinearSVC 의 정답률 :  0.9
LogisticRegression 의 정답률 :  0.9333333333333333
LogisticRegressionCV 의 정답률 :  0.9333333333333333
MLPClassifier 의 정답률 :  0.9333333333333333
MultinomialNB 의 정답률 :  0.9333333333333333
NearestCentroid 의 정답률 :  0.9
NuSVC 의 정답률 :  0.9666666666666667
PassiveAggressiveClassifier 의 정답률 :  0.8333333333333334
Perceptron 의 정답률 :  0.7666666666666667
QuadraticDiscriminantAnalysis 의 정답률 :  0.9666666666666667
RadiusNeighborsClassifier 의 정답률 :  0.9666666666666667
RandomForestClassifier 의 정답률 :  0.9333333333333333
RidgeClassifier 의 정답률 :  0.7666666666666667
RidgeClassifierCV 의 정답률 :  0.7666666666666667
SGDClassifier 의 정답률 :  0.8
SVC 의 정답률 :  0.9666666666666667
'''





