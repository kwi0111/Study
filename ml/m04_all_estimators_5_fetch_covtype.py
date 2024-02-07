from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    train_size=0.9,
                                                    random_state=123,
                                                    stratify=y,
                                                    shuffle=True
                                                    ) 

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
        continue
    
    
    
    
    
'''
#3.컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test) 
y_predict = model.predict(x_test)
 
acc = accuracy_score(y_test, y_predict)
print("acc : ", results)

# LinearSVC                     
# Perceptron                  acc :  0.5297580117724002
# LogisticRegression           acc :  0.6179477470655055
# KNeighborsClassifier        acc :  0.9706722660149393
# DecisionTreeClassifier      acc :  0.9428246876183264
# RandomForestClassifier        acc :  0.9572648101614403

'''