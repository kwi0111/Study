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

class CustomXGBClassifier(XGBClassifier):
    def __str__(self):
        return 'XGBClassifier()'

#1. 데이터
path = "c:\\_data\\dacon\\diabetes\\"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path +"test.csv",index_col=0  ).drop(['Pregnancies', 'DiabetesPedigreeFunction'], axis=1)
test = train_csv['SkinThickness']
for i in range(test.size):
      if test[i] == 0:
         test[i] =test.mean()
        
#train_csv['BloodPressure'] = test

submission_csv = pd.read_csv(path + "sample_submission.csv") 

print(train_csv.columns) #'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
      # 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
      
x = train_csv.drop(['Outcome','Pregnancies','DiabetesPedigreeFunction'], axis=1)
y = train_csv['Outcome']


from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델 구성
from xgboost import XGBClassifier
model1 = DecisionTreeClassifier(random_state=777)
model2 = RandomForestClassifier(random_state=777)
model3 = GradientBoostingClassifier(random_state=777)
model4 = CustomXGBClassifier(random_state=777, cv=kfold)
# model4 = XGBClassifier(random_state=777, cv=kfold)


models = [model1, model2, model3, model4]

for model in models :
    model.fit(x_train, y_train)
    print("========================", model, '=======================')
    print(model)
    print("acc : ", model.score(x_test, y_test))
    print(model.feature_importances_)