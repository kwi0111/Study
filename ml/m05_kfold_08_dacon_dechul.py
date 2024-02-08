import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


#1. 데이터
path = "C:\\_data\\dacon\\dechul\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0 )
print(train_csv.shape)  # (96294, 14)
test_csv = pd.read_csv(path + "test.csv", index_col=0 )
print(test_csv.shape)  # (64197, 13)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv.shape)  # (64197, 2)


# 라벨 엔코더
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder() # 대출기간, 대출목적, 근로기간, 주택소유상태 // 라벨 인코더 : 카테고리형 피처를 숫자형으로 변환
train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = le.fit_transform(train_csv['대출목적'])
train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])

test_csv['주택소유상태'] = le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = le.fit_transform(test_csv['대출목적'])
test_csv['대출기간'] = test_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
test_csv['근로기간'] = le.fit_transform(test_csv['근로기간'])

train_csv['대출등급'] = le.fit_transform(train_csv['대출등급']) # 마지막에 와야함

# x와 y를 분리
x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

y = np.reshape(y, (-1,1)) 

x_train, x_test, y_train, y_test = train_test_split(
                                                    x,
                                                    y,             
                                                    train_size=0.86,
                                                    random_state=2024,
                                                    stratify=y,
                                                    shuffle=True,
                                                    )
n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
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


#3.컴파일, 훈련
scores = cross_val_score(model, x_train, y_train, cv = kfold)
print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores,), 4))
'''
model.fit(x_train, y_train)


#4.평가, 예측
results = model.score(x_test, y_test) 
y_predict = model.predict(x_test)
 
acc = accuracy_score(y_test, y_predict)
print("acc : ", results)
# acc :  0.33785788458685656


# LinearSVC                     0.31004302032339415
# Perceptron                    0.2939474855362706
# LogisticRegression           0.3572912030855956
# KNeighborsClassifier        0.4320575582257825
# DecisionTreeClassifier      0.8373386737872719
# RandomForestClassifier        0.8078920041536864
'''


'''

'''


