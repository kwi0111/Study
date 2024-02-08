import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

#1. 데이터
# x, y = load_iris(return_X_y=True)
datasets = load_iris()
# x = datasets.data
# y = datasets.target
df = pd.DataFrame(datasets.data, columns = datasets.feature_names)
print(df)   # [150 rows x 4 columns]
n_splits = 3
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)   # 정의만 내렸다.
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)   # y값 있어야함

# 어떻게 잘렸나 확인
for train_index, val_index in kfold.split(df):      
    print('+'*50)       
    print(train_index, "\n", val_index)
    print("훈련데이터 갯수 :", len(train_index), " ",
          "검증데이터 갯수 :", len(val_index))


'''

#2. 모델
model = SVC()

#3. 훈련
scores = cross_val_score(model, x, y, cv = kfold)
print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores,), 4)) # ACC :  [0.96666667 0.96666667 1.         0.96666667 0.93333333] 5분할 했으니까 5개 나옴.

'''

#4.

'''
kFold
평균 ACC :  0.96

RandomForestClassifier
accuracy_score :  0.93
'''

