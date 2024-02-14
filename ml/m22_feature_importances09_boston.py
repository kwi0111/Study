from sklearn.datasets import load_boston

# 현재 사이킷런 버전 1.3.0 보스턴 안됨. 삭제 후 재설치
#pip uninstall scikit-learn
#pip uninstall scikit-image                  
#pip uninstall scikit-learn-intelex
#pip install scikit-learn==1.1.3


datasets = load_boston()
print(datasets)
x = datasets.data
y = datasets.target
print(x)
print(x.shape) # (506, 13)
print(y.shape) # (506,)

print(datasets.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
# 'B' 'LSTAT']

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore') #warning 무시. 나타내지않음.
import time
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV


#1. 데이터
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델구성
from xgboost import XGBRegressor
model = XGBRegressor(cv=5, verbose=1,)

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

from sklearn.metrics import r2_score
print('score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score :", r2_score(y_test, y_predict))
print("최적튠 ACC :", r2_score(y_test, y_predict))
print("걸린시간 :", round(end_time - start_time, 2), "초")
print (type(model).__name__, model.feature_importances_)

