from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
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

#2. 모델구성
allAlgorithms = all_estimators(type_filter='classifier')

for name, algorithm in allAlgorithms:
    try:
        #2. 모델
        model = algorithm()
        #.3 훈련
        model.fit(x_train, y_train)
        
        acc = model.score(x_test, y_test)   
        print(name, "의 정답률 : ", round(acc, 2))   
    except: 
        continue    # 그냥 다음껄로 넘어간다.
'''
#3.컴파일, 훈련
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















 




