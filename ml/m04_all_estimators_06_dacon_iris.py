# https://dacon.io/competitions/open/236070/overview/description
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

#2. 모델 구성 
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
# .3 컴파일, 훈련
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