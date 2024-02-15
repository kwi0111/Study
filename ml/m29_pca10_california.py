import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time
from sklearn.svm import LinearSVR
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV
import pandas as pd

#1. 데이터
datasets = fetch_california_housing()
x = np.array(datasets.data)
y = np.array(datasets.target)

# ''' 25퍼 미만 열 삭제 '''
# columns = datasets.feature_names
# # columns = x.columns
# x = pd.DataFrame(x,columns=columns)
# print("x.shape",x.shape)
# # ''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
# fi_str = "0.47826383 0.07366086 0.0509511  0.02446287 0.02366972 0.14824368\
#  0.0921493  0.10859864"
 
# ''' str에서 숫자로 변환하는 구간 '''
# fi_str = fi_str.split()
# fi_float = [float(s) for s in fi_str]
# print(fi_float)
# fi_list = pd.Series(fi_float)

# ''' 25퍼 미만 인덱스 구하기 '''
# low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
# print('low_idx_list',low_idx_list)

# ''' 25퍼 미만 제거하기 '''
# low_col_list = [x.columns[index] for index in low_idx_list]
# # 이건 혹여 중복되는 값들이 많아 25퍼이상으로 넘어갈시 25퍼로 자르기
# if len(low_col_list) > len(x.columns) * 0.25:   
#     low_col_list = low_col_list[:int(len(x.columns)*0.25)]
# print('low_col_list',low_col_list)
# x.drop(low_col_list,axis=1,inplace=True)
# print("after x.shape",x.shape)

scaler = StandardScaler()
x = scaler.fit_transform(x)

from sklearn.decomposition import PCA
n_components = 7
pca = PCA(n_components=n_components)
x = pca.fit_transform(x) 

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8)

#2. 모델구성
for i in range(x.shape[1], 0, -1):
    # pca = PCA(n_components=i)
    # x_train = pca.fit_transform(x_train)
    # x_test = pca.transform(x_test)
    
    # 모델 초기화
    model = RandomForestRegressor()
    
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