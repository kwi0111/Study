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


'''

