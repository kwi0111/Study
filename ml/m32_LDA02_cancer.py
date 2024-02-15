import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline  # 파이프라인 = 일괄 처리 / 함수
from sklearn.pipeline import Pipeline  # 클래스 /
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier


#1.데이터
datasets = load_breast_cancer()

x = datasets.data
y= datasets.target

print(x.shape, y.shape)
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357], dtype=int64))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=1) # 이진 분류는 0과 1이니까 2 - 1 // 1보다 클수 없다.
# n_components는  n_features 또는 n_classes -1 값보다는 커야한다.
lda.fit(x,y)
x = lda.transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)

 
#2. 모델 구성
for i in range(x.shape[1], 0, -1):
    # pca = PCA(n_components=i)
    # x_train = pca.fit_transform(x_train)
    # x_test = pca.transform(x_test)
    
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

evr = lda.explained_variance_ratio_
print(evr)
print(sum(evr))

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)