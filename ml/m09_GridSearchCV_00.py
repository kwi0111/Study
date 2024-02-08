'''
하이퍼 파라미터의 자동화
svc (C) C를 다돌려 본다.
감마도 다 돌려 버리자.
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, train_size=0.8, stratify=y)

n_splits = 3
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        model = SVC(gamma=gamma, C=C)
        model.fit(x_train, y_train)
        
        score = model.score(x_test, y_test)
        
        if score > best_score:
            best_score = score
            best_parameters = {'C':C, 'gamma':gamma}

print("최고점수 {:2f} ".format(best_score))
print("최적의 매개변수 : ", best_parameters)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)   # 정의만 내렸다. // 하이퍼파라미터의 자동화
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)   # 분류 모델


'''
#2. 모델
model = SVC()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv = kfold)
print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores,), 4)) # ACC :  [0.96666667 0.96666667 1.         0.96666667 0.93333333] 5분할 했으니까 5개 나옴.

y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
print(y_predict) # cv의 예측치
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC :' ,acc)

#4.

kFold
평균 ACC :  0.96

RandomForestClassifier
accuracy_score :  0.93

cross_val_predict ACC : 0.9333333333333333
'''
'''

'''




