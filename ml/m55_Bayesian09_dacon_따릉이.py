
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression # 앤 분류다
from sklearn.ensemble import StackingRegressor
from catboost import CatBoostRegressor
from bayes_opt import BayesianOptimization
import time

#1. 데이터
path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0) # \ \\ / // 다 가능( 예약어 사용할때 두개씩 사용) 인덱스컬럼은 0번째 컬럼이다라는뜻.
test_csv = pd.read_csv(path +"test.csv", index_col=0)
submission_csv = pd.read_csv(path + "submission.csv") 

print(train_csv.info())
train_csv = train_csv.fillna(train_csv.mean())  #결측치가 하나라도 있으면 행전체 삭제됨.
test_csv = test_csv.fillna(test_csv.mean())   # (0,mean)

# test_csv = test_csv.drop(['hour_bef_humidity','hour_bef_windspeed'], axis=1)   # (0,mean)

print(train_csv.shape)      #(1328, 10)

################# x와 y를 분리 ###########
x = train_csv.drop(['count',], axis=1)
y = train_csv['count']


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size= 0.8)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
bayesian_params = {
    'learning_rate': (0.001, 1),
    'max_depth': (3, 10),
    'num_leaves': (24, 40),
    'min_child_samples': (10, 200),
    'min_child_weight': (1, 50),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1),
    'max_bin': (9, 500),
    'reg_lambda': (0.001, 10),
    'reg_alpha': (0.01, 50),
}

def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators': 100,
        'learning_rate': learning_rate,
        'max_depth': int(round(max_depth)),
        'num_leaves': int(round(num_leaves)),
        'min_child_samples': int(round(min_child_samples)),
        'min_child_weight': int(round(min_child_weight)),
        'subsample': max(min(subsample, 1), 0),
        'colsample_bytree': colsample_bytree,
        'max_bin': max(int(round(max_bin)), 10),
        'reg_lambda': max(reg_lambda, 0),
        'reg_alpha': reg_alpha,
        'objective': 'reg:squarederror',  
    }
    model = XGBRegressor(**params, n_jops=-1)
    model.fit(x_train, y_train,)
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    return results

bay = BayesianOptimization(
    f = xgb_hamsu,
    pbounds=bayesian_params,
    random_state=456,
)

n_iter = 200
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(bay.max)
print(n_iter, '번 걸린시간 : ', round(end_time-start_time, 2), '초')

##########################

# r2_score : 0.7341018017406806 -> n_estimators=100으로 r2_score : 0.7381181281326573
# 100, False 최종점수 0.7514021768862288

# 보팅 regressor r2_score : 0.7302023854843464

# 스테킹 cv5 r2_score : 0.7565754794531557
# 스테킹 cv3  r2_score : 0.7486647585934258

'''
{'target': 0.7786940833121438, 
'params': {'colsample_bytree': 0.9480409408809098, 'learning_rate': 0.24393746405607472, 'max_bin': 366.3104843952526, 
'max_depth': 8.87249027726861, 'min_child_samples': 29.78862785277498, 'min_child_weight': 6.102719544371601, 'num_leaves': 37.14406482603165, 
'reg_alpha': 47.252834070513366, 'reg_lambda': 1.3048338903283232, 'subsample': 0.8801571375382964}}
200 번 걸린시간 :  41.16 초
'''
















