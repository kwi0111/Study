# 랜덤포레스트의 알고리즘이 배깅이다.
# 배깅 - 보팅
# votiong - 모델 여러개 / 같은 데이터 / 소프트 or 하드 / 하드는 다수결 / 소프트는 평균에서 높은쪽
# bagging - 모델 1개 - 데이터가 다르다 (샘플링해서) // 에포 느낌 = n이스터메이트 //  중복 된다 // 

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression # 앤 분류다
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization
import time


#1. 데이터
path = 'C:\\_data\\kaggle\\Obesity_Risk\\'
train_csv=pd.read_csv(path+"train.csv",index_col=0)
test_csv=pd.read_csv(path+"test.csv",index_col=0)
submission_csv=pd.read_csv(path+"sample_submission.csv")
# x = train_csv.drop(['NObeyesdad'], axis=1)
# y = train_csv['NObeyesdad']



# 훈련 데이터와 테스트 데이터 합치기
# combined_data = pd.concat([x, test_csv])


##############데이터 전처리###############


def perform_feature_engineering(df):

    train_csv['BMI'] = train_csv['Weight'] / (train_csv['Height'] ** 2)
    test_csv['BMI'] = test_csv['Weight'] / (test_csv['Height'] ** 2)

    train_csv['bmioncp'] = train_csv['BMI'] / train_csv['NCP']
    test_csv['bmioncp'] = test_csv['BMI'] / test_csv['NCP']

    train_csv['WIR'] = train_csv['Weight'] / train_csv['CH2O']
    test_csv['WIR'] = test_csv['Weight'] / test_csv['CH2O']
    
    train_csv = perform_feature_engineering(train_csv)
    test_csv = perform_feature_engineering(test_csv)
    return df


#Gender
train_csv['Gender']= train_csv['Gender'].str.replace("Male","0")
train_csv['Gender']= train_csv['Gender'].str.replace("Female","1")
test_csv['Gender']= test_csv['Gender'].str.replace("Male","0")
test_csv['Gender']= test_csv['Gender'].str.replace("Female","1")

# print(train_csv['Gender'])
# print(test_csv['Gender'])



#family_history_with_overweight
train_csv['family_history_with_overweight']= train_csv['family_history_with_overweight'].str.replace("yes","0")
train_csv['family_history_with_overweight']= train_csv['family_history_with_overweight'].str.replace("no","1")
test_csv['family_history_with_overweight']= test_csv['family_history_with_overweight'].str.replace("yes","0")
test_csv['family_history_with_overweight']= test_csv['family_history_with_overweight'].str.replace("no","1")

# print(train_csv['family_history_with_overweight'])
# print(test_csv['family_history_with_overweight'])

train_csv['FAVC']= train_csv['FAVC'].str.replace("yes","0")
train_csv['FAVC']= train_csv['FAVC'].str.replace("no","1")
test_csv['FAVC']= test_csv['FAVC'].str.replace("yes","0")
test_csv['FAVC']= test_csv['FAVC'].str.replace("no","1")

#print(train_csv['FAVC'])
#print(test_csv['FAVC'])
#print(np.unique(train_csv['FAVC'], return_counts= True))
#print(np.unique(test_csv['FAVC'], return_counts= True))


#print(np.unique(train_csv['CAEC'], return_counts= True))
train_csv['CAEC']= train_csv['CAEC'].str.replace("Always","0")
train_csv['CAEC']= train_csv['CAEC'].str.replace("Frequently","1")
train_csv['CAEC']= train_csv['CAEC'].str.replace("Sometimes","2")
train_csv['CAEC']= train_csv['CAEC'].str.replace("no","3")

test_csv['CAEC']= test_csv['CAEC'].str.replace("Always","0")
test_csv['CAEC']= test_csv['CAEC'].str.replace("Frequently","1")
test_csv['CAEC']= test_csv['CAEC'].str.replace("Sometimes","2")
test_csv['CAEC']= test_csv['CAEC'].str.replace("no","3")
#print(np.unique(train_csv['CAEC'], return_counts= True))
#print(np.unique(test_csv['CAEC'], return_counts= True))


#print(np.unique(test_csv['SMOKE'], return_counts= True))
train_csv['SMOKE']= train_csv['SMOKE'].str.replace("yes","0")
train_csv['SMOKE']= train_csv['SMOKE'].str.replace("no","1")
test_csv['SMOKE']= test_csv['SMOKE'].str.replace("yes","0")
test_csv['SMOKE']= test_csv['SMOKE'].str.replace("no","1")

#print(np.unique(train_csv['SMOKE'], return_counts= True))
#print(np.unique(test_csv['SMOKE'], return_counts= True))

#print(np.unique(train_csv['SCC'], return_counts= True))
train_csv['SCC']= train_csv['SCC'].str.replace("yes","0")
train_csv['SCC']= train_csv['SCC'].str.replace("no","1")
test_csv['SCC']= test_csv['SCC'].str.replace("yes","0")
test_csv['SCC']= test_csv['SCC'].str.replace("no","1")
#print(np.unique(test_csv['SCC'], return_counts= True))


#print(np.unique(test_csv['CALC'], return_counts= True))
test_csv['CALC']= test_csv['CALC'].str.replace("Always","1")
test_csv['CALC']= test_csv['CALC'].str.replace("Frequently","1")
test_csv['CALC']= test_csv['CALC'].str.replace("Sometimes","2")
test_csv['CALC']= test_csv['CALC'].str.replace("no","3")

#print(np.unique(train_csv['CALC'], return_counts= True))
train_csv['CALC']= train_csv['CALC'].str.replace("Always","0")
train_csv['CALC']= train_csv['CALC'].str.replace("Frequently","1")
train_csv['CALC']= train_csv['CALC'].str.replace("Sometimes","2")
train_csv['CALC']= train_csv['CALC'].str.replace("no","3")
#print(np.unique(train_csv['CALC'], return_counts= True))


#print(np.unique(train_csv['MTRANS'], return_counts= True))
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Automobile","0")
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Bike","1")
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Motorbike","2")
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Public_Transportation","3")
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Walking","4")

test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Automobile","0")
test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Bike","1")
test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Motorbike","2")
test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Public_Transportation","3")
test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Walking","4")


x = train_csv.drop(['NObeyesdad', 'SMOKE'], axis = 1)
y = train_csv['NObeyesdad']
test_csv = test_csv.drop(['SMOKE'], axis = 1)

le = LabelEncoder()
y = le.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size= 0.8, stratify=y)

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
    }
    model = XGBClassifier(**params, n_jops=-1)
    model.fit(x_train, y_train,
              eval_set = [(x_train,y_train), (x_test, y_test)],
              eval_metric = 'mlogloss',
              verbose = 0,
              early_stopping_rounds=50,
              )
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
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
# 부트스트랩 True acc_score : 0.8131021194605009 -> n_estimators 100으로 acc_score : 0.8157514450867052
# 부트스트랩 False acc_score : 0.8128612716763006

# 보팅 하드 acc_score : 0.8954720616570327
# 보팅 소프트 acc_score : 0.9024566473988439

# 스테킹 cv = 5 acc_score : 0.8986030828516378
# 스테킹 cv = 3 acc_score : 0.9002890173410405
'''
{'target': 0.8580403966976479, 
'params': {'colsample_bytree': 0.8321552943522509, 'learning_rate': 0.22445408048097323, 'max_bin': 459.52552929351197, 
'max_depth': 8.979460639298134, 'min_child_samples': 134.65862671259296, 'min_child_weight': 5.594231273044766, 'num_leaves': 34.46903970964884, 
'reg_alpha': 1.8662117067878878, 'reg_lambda': 6.6475208557865955, 'subsample': 0.8679650848325389}}        
200 번 걸린시간 :  299.93 초
'''
















