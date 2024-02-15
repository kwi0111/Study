from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC #softvector machine
from sklearn.linear_model import Perceptron, LogisticRegression , LinearRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV


#1. 데이터

datasets = load_wine()
print(datasets.DESCR) 
print(datasets.feature_names) #'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'       


x= datasets.data
y= datasets.target

scaler = StandardScaler()
x = scaler.fit_transform(x)

print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
n_components = 2
lda = LinearDiscriminantAnalysis(n_components=n_components) #  n_classes(라벨) 의 갯수에 영향을 받는다. // 차원이 줄어든다. // 분류 부분만 쓴다.
# n_components는  n_features 또는 n_classes -1 값보다는 커야한다.
lda.fit(x,y)
x = lda.transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)


#2. 모델구성
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

# score : 0.9444444444444444
# accuracy_score : 0.9444444444444444
# 최적튠 ACC : 0.9444444444444444
# 걸린시간 : 0.07 초
# XGBClassifier [0.44280764 0.47587222 0.04364977 0.03767039]

# score : 0.9444444444444444
# accuracy_score : 0.9444444444444444
# 최적튠 ACC : 0.9444444444444444
# 걸린시간 : 0.05 초
# XGBClassifier [0.44280764 0.47587222 0.04364977 0.03767039]
