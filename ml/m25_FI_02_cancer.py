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

class CustomXGBClassifier(XGBClassifier):
    def __str__(self):
        return 'XGBClassifier()'

#1.데이터
datasets = load_breast_cancer()

x = datasets.data
y= datasets.target

# ### 넘파이 삭제
# x = np.delete(x, 0, axis=1)
# x = np.delete(x, slice(1,5), axis=1)
# print(x)  


# # # ### 판다스 삭제
# ''' 25퍼 미만 열 삭제 '''
# columns = datasets.feature_names
# # columns = x.columns
# x = pd.DataFrame(x,columns=columns)
# print("x.shape",x.shape)
# # ''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
# fi_str = "0.01815764 0.01569501 0.         0.0032644  0.00396126 0.00207543\
#  0.00426753 0.06830692 0.00221316 0.00328998 0.01382377 0.00926016\
#  0.0153579  0.01026187 0.0181306  0.00254898 0.         0.01395431\
#  0.00322362 0.00251621 0.00386465 0.02203292 0.6079987  0.05683359\
#  0.01485889 0.00435244 0.01172198 0.05729257 0.00671581 0.00401975"
 
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

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
 
#2. 모델 구성
model1 = DecisionTreeClassifier(random_state=777)
model2 = RandomForestClassifier(random_state=777)
model3 = GradientBoostingClassifier(random_state=777)
model4 = CustomXGBClassifier(random_state=777, cv=kfold)

models = [model1, model2, model3, model4]

for model in models :
    model.fit(x_train, y_train)
    print("========================", model, '=======================')
    print(model)
    print("acc : ", model.score(x_test, y_test))
    print(model.feature_importances_)
    

