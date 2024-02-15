#https://dacon.io/competitions/open/236068/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC #softvector machine
from sklearn.linear_model import Perceptron, LogisticRegression , LinearRegression#분류!
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor,ExtraTreeClassifier
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import Pipeline  # 클래스 /
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#1. 데이터
path = "c:\\_data\\dacon\\diabetes\\"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path +"test.csv",index_col=0  ).drop(['Pregnancies', 'DiabetesPedigreeFunction'], axis=1)
test = train_csv['SkinThickness']
for i in range(test.size):
      if test[i] == 0:
         test[i] =test.mean()
submission_csv = pd.read_csv(path + "sample_submission.csv") 

print(train_csv.columns) #'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
      # 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
      

x = train_csv.drop(['Outcome','Pregnancies','DiabetesPedigreeFunction'], axis=1)
y = train_csv['Outcome']

from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

from sklearn.decomposition import PCA
# n_components = 5
# pca = PCA(n_components=n_components)
# x = pca.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)

#2. 모델 구성
for i in range(x.shape[1], 0, -1):
    pca = PCA(n_components=i)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    
    # 모델 초기화
    model = RandomForestClassifier()
    
    # 모델 훈련
    model.fit(x_train, y_train)
    
    # 모델 평가
    results = model.score(x_test, y_test)
    print('====================================')
    print(x_train.shape)
    print('model.score : ', results)
    break
    
evr = pca.explained_variance_ratio_
print(evr)
print(sum(evr))

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

# ======================== XGBClassifier() =======================
# XGBClassifier()
# acc :  0.7404580152671756
# [0.3384777  0.13798998 0.14578097 0.20828696 0.16946442]

# PCA
# ======================== XGBClassifier() =======================
# XGBClassifier()
# acc :  0.6793893129770993
# [0.38759443 0.2119237  0.16608316 0.23439874]

