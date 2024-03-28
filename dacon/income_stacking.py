import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, Normalizer
from sklearn.experimental import enable_halving_search_cv #정식버전이 아님!
from sklearn.model_selection import train_test_split, HalvingGridSearchCV, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor, GradientBoostingRegressor
from keras.callbacks import ReduceLROnPlateau
import optuna
from catboost import CatBoostRegressor
import pickle
import time
import random


random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)


#1. 데이터

path = 'C:\\_data\\dacon\\income\\'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
#print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
#print(test_csv)
submission_csv = pd.read_csv(path + 'sample_submission.csv')
#print(submission_csv)

# print(train_csv.shape) #(20000, 22)
# print(test_csv.shape) #(10000, 21)
# print(submission_csv.shape) #(10000, 2)

#print(train_csv.columns)
# ['Age', 'Gender', 'Education_Status', 'Employment_Status',
#        'Working_Week (Yearly)', 'Industry_Status', 'Occupation_Status', 'Race',
#        'Hispanic_Origin', 'Martial_Status', 'Household_Status',
#        'Household_Summary', 'Citizenship', 'Birth_Country',
#        'Birth_Country (Father)', 'Birth_Country (Mother)', 'Tax_Status',
#        'Gains', 'Losses', 'Dividends', 'Income_Status', 'Income'],

#print(test_csv.isnull().sum()) #1개
#print(train_csv.isnull().sum()) #없음.

test_csv = test_csv.fillna(method= 'bfill')



##########Citizen 값 병합#################
train_csv['Citizenship'] = train_csv['Citizenship'].apply(lambda x: 'Native' if 'Native' in x else x)
test_csv['Citizenship'] = test_csv['Citizenship'].apply(lambda x: 'Native' if 'Native' in x else x)
# print(np.unique(train_csv['Citizenship'], return_counts= True))

###########Birth_Country 병합############
#print(np.unique(train_csv['Birth_Country'], return_counts= True))

# train_csv.loc[train_csv['Birth_Country'] != 'US', 'Birth_Country'] = 'not US'
# train_csv.loc[train_csv['Birth_Country (Father)'] != 'US', 'Birth_Country (Father)'] = 'not US'
# train_csv.loc[train_csv['Birth_Country (Mother)'] != 'US', 'Birth_Country (Mother)'] = 'not US'

# test_csv.loc[test_csv['Birth_Country'] != 'US', 'Birth_Country'] = 'not US'
# test_csv.loc[test_csv['Birth_Country (Father)'] != 'US', 'Birth_Country (Father)'] = 'not US'
# test_csv.loc[test_csv['Birth_Country (Mother)'] != 'US', 'Birth_Country (Mother)'] = 'not US'


label_encoder_dict = {}
for label in train_csv:
    data = train_csv[label].copy()
    if data.dtypes == 'object':
        label_encoder = LabelEncoder()
        train_csv[label] = label_encoder.fit_transform(data)
        label_encoder_dict[label] = label_encoder
# print(train_csv.head(10))

# # inverse_transform 작동 확인 완료
# for label, label_encoder in label_encoder_dict.items():
#     data = train_csv[label].copy()
#     train_csv[label] = label_encoder.inverse_transform(data)
# print(train_csv.head(10))


for label, encoder in label_encoder_dict.items():
    data = test_csv[label]
    test_csv[label] = encoder.transform(data)
#print(test_csv.head(10))

# print(test_csv.isna().sum())

# 삭제할 컬럼
# Household_Summary
# Income_Status

# x = train_csv.drop(['Income'], axis=1)
x = train_csv.drop(['Household_Summary','Income'], axis=1)
y = train_csv['Income']
test_csv = test_csv.drop(['Household_Summary'], axis=1)





from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8,  shuffle= True, random_state= 1113)


#3. 훈련

xgb_params = {
    'n_estimators': 2000,  # 조금 증가
    'max_depth': 9,  # 깊이 증가
    'min_child_weight': 10,
    'gamma': 2.5,  # 소폭 조정
    'learning_rate': 0.005,  # 학습률 증가
    'colsample_bytree': 0.45,  # 소폭 증가
    'lambda': 3,  # 레귤러리제이션 조정
    'alpha': 1,  # 레귤러리제이션 조정
    'subsample': 0.7  # 소폭 증가
}

lgbm_params = {
    'n_estimators': 800,  # 조금 증가
    'max_depth': 13,  # 깊이 증가
    'min_child_weight': 3,
    'gamma': 0.07,  # 소폭 조정
    'learning_rate': 0.007,  # 학습률 증가
    'colsample_bytree': 0.55,  # 소폭 증가
    'lambda': 1,  # 레귤러리제이션 조정
    'alpha': 5,  # 레귤러리제이션 증가
    'subsample': 0.85  # 유지
}


xgb = XGBRegressor(**xgb_params)
lgb = LGBMRegressor(**lgbm_params)
rf = RandomForestRegressor()
# gbr = GradientBoostingRegressor()

model = VotingRegressor(
     estimators=[
                ('LGBM',lgb),
                  ('RF',rf),
                 ('XGB',xgb),
                #  ('GBR', gbr)
                 ],
     
    # final_estimator=CatBoostRegressor(verbose=1),
    n_jobs= -1,
    # cv=8, 
    verbose=1
)
                    
#3. 훈련

model.fit(x_train, y_train)

#4.  평가, 예측
                    
 
print("=====================================")
y_submit = model.predict(test_csv)
submission_csv['Income'] = y_submit


r2 = model.score(x_test,y_test)
pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(pred,y_test))
y_predict = model.predict(x_test)
score = r2_score(y_test, y_predict)

import datetime
dt = datetime.datetime.now()
y_submit = model.predict(test_csv)
submission_csv['Income'] = y_submit
submission_csv.to_csv(path+f'submit_{dt.day}day{dt.hour:2}{dt.minute:2}_rmse_{rmse:4f}.csv',index=False)
print("R2:   ",r2)
print("RMSE: ",rmse)


pickle.dump(model, open(path + f'submit_{dt.day}day{dt.hour:2}{dt.minute:2}rmse{rmse:4}.dat', 'wb'))


