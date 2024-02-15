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

print(x.shape, y.shape) # (569, 30) (569,)
scaler = StandardScaler()
x = scaler.fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA() # n_components 없으면 최대값 디폴트
x = pca.fit_transform(x) 
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)
print('===========')

print(x.shape[0], x.shape[1]) # x.shape[0] : 행의 수 // x.shape[1] : 열의 수
#2. 모델 구성
for i in range(x.shape[1], 0, -1):  # x의 열의 수부터 시작하여 1씩 감소하면서 0까지 반복하는 반복문
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
# model.score :  0.9473684210526315
evr = pca.explained_variance_ratio_
print(evr)
print(sum(evr))
#  explained_variance_ratio_ : 전체 데이터 분산의 얼마를 설명하는지를 나타내며, 이 값이 높을수록 해당 주성분이 중요한 정보를 많이 담고 있음을 의미
evr_cumsum = np.cumsum(evr) # 주어진 배열의 각 원소를 순차적으로 더한 결과를 반환
print(evr_cumsum)