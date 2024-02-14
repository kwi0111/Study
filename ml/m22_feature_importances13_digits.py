from sklearn.datasets import load_digits
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time

datasets = load_digits()

x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape) # (1797, 64) (1797,)
print(pd.value_counts(y, sort=False))
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180
x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        train_size=0.7,
        random_state=200,    
        stratify=y,
        shuffle=True,
        )
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
#2. 모델구성
from xgboost import XGBClassifier
model = XGBClassifier(cv=5, verbose=1,)

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

from sklearn.metrics import accuracy_score
print('score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))
print("최적튠 ACC :", accuracy_score(y_test, y_predict))
print("걸린시간 :", round(end_time - start_time, 2), "초")
print (type(model).__name__, model.feature_importances_)

