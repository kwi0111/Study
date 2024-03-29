from keras.preprocessing.text import Tokenizer
import os
import random as rn
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv1D, LSTM, Flatten, Embedding, Input, Concatenate, concatenate, Reshape
from sklearn.metrics import r2_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split,  GridSearchCV, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, MinMaxScaler, Normalizer, RobustScaler, StandardScaler, MaxAbsScaler
from keras.utils import to_categorical
import warnings
warnings.filterwarnings(action='ignore')
le = LabelEncoder()
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline,Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

SEED1 = 43
SEED2 = 43
SEED3 = 43

tf.random.set_seed(SEED1)  
np.random.seed(SEED2)
rn.seed(SEED3)



#1. 데이터

path = 'C:\\_data\\kaggle\\Obesity_Risk\\'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

#print(train_csv)
#print(test_csv)
#print(submission_csv)

# print(train_csv.shape) #(20758, 17)
# print(test_csv.shape) #(13840, 16)
# print(submission_csv.shape) #(13840, 2)

#print(train_csv.columns) #Index(['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
#       'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
#       'CALC', 'MTRANS', 'NObeyesdad'],



#print(train_csv['NObeyesdad'].value_counts())
# Obesity_Type_III       4046
# Obesity_Type_II        3248
# Normal_Weight          3082
# Obesity_Type_I         2910
# Insufficient_Weight    2523
# Overweight_Level_II    2522
# Overweight_Level_I     2427


##############데이터 전처리###############
le = LabelEncoder()


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

#print(np.unique(train_csv['MTRANS'], return_counts= True))
#print(np.unique(test_csv['MTRANS'], return_counts= True))



#print(test_csv.isnull().sum()) #없음.
#print(train_csv.isnull().sum()) #없음.


x = train_csv.drop('NObeyesdad', axis = 1)
y = train_csv['NObeyesdad']

y = le.fit_transform(y)

# y = y.values.reshape(-1,1)
# ohe = OneHotEncoder(sparse = False)
# ohe.fit(y)
# y = ohe.transform(y)
# print(y.shape)  

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
import time


#mms = MinMaxScaler()
# mms = MinMaxScaler(feature_range=(0,1))
#mms = StandardScaler()
#mms = MaxAbsScaler()
#mms = RobustScaler()

# mms.fit(x)
# x = mms.transform(x)
# test_csv=mms.transform(test_csv)
#print(x.shape, y.shape)  #(20758, 16) (20758,)
#print(np.unique(y, return_counts= True))
#(array([0, 1, 2, 3, 4, 5, 6]), array([2523, 3082, 2910, 3248, 4046, 2427, 2522], dtype=int64))




x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.9, shuffle=True, random_state=5, stratify= y)
#5 9158 19 9145
n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# y_train = to_categorical(y_train, 7)
# y_test = to_categorical(y_test, 7) 






# columns_id = np.arange(0, x.shape[1])
# categorical_features_id = columns_id[x.dtypes == object]

# mms = MinMaxScaler() #(feature_range=(0,1))
# mms = StandardScaler()
mms = MaxAbsScaler()
# mms = RobustScaler()
#mms = Normalizer()


mms.fit(x_train)
x_train= mms.transform(x_train)
x_test= mms.transform(x_test)
test_csv= mms.transform(test_csv)


#==================================
#2. 모델
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#model = RandomForestClassifier(max_depth=100, random_state=1117)#1001 #1117

################lgbm########################
lgb_params = {
    "objective": "multiclass",          
    "metric": "multi_logloss",          
    "verbosity": 1,                    
    "boosting_type": "gbdt",            
    "random_state": 42,       
    "num_class": 7,                     
    'learning_rate': 0.030962211546832760,  
    'n_estimators': 20000,                
    'reg_alpha' : 0.9269816785,
    'max_depth': 100,                    
    'colsample_bytree': 0.40977129346872643,  
    'subsample': 0.535797422450176,    
    'n_jobs' : -1,
    'min_child_samples': 10             
}


model = LGBMClassifier(**lgb_params)
################lgbm########################


################xgb########################

# params = {
#     'n_estimators': 30000,
#     'learning_rate': 0.12,
#     'gamma': 0.8024196354156454324,
#     'reg_alpha': 0.9025931173755949,
#     'reg_lambda': 0.96835667255875388,
#     'max_depth': 50,
#     'min_child_weight': 20,
#     'subsample': 0.893274050086088,
#     'colsample_bytree': 0.7579828557036317,
# }

# model = XGBClassifier(**params, random_state = 28233)
################xgb########################

################catboost########################
# model = CatBoostClassifier(auto_class_weights = 'Balanced', 
#                            iterations=50000,
#                            learning_rate=0.002162645,
#                            max_depth=16,
#                            l2_leaf_reg=18,
#                            max_bin=15,
#                            early_stopping_rounds=70, 
#                            random_state=98765)


# param = {'auto_class_weights':'Balanced','iterations':10000,'early_stopping_rounds':100,'l2_leaf_reg':18,'learning_rate':0.0021626,
#          'max_depth':16,'task_type':'GPU','max_bin':8, 'random_state':9876}
# model = CatBoostClassifier(**param, bootstrap_type='Poisson')

################catboost########################


#3. 컴파일 , 훈련


x_train = np.asarray(x_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
test_csv = np.asarray(test_csv).astype(np.float32)

start_time = time.time()
model.fit(x_train, y_train)#, eval_set=(x_test, y_test))#, cat_features=categorical_features_id)
end_time = time.time()

#4. 평가, 예측
# print("최적의 매개변수 : ", model.best_estimator_)
# print("최적의 파라미터 : ", model.best_params_)
# print('best_score :', model.best_score_)
print('score :', model.score(x_test, y_test))


results = model.score(x_test, y_test)
print("acc :", results)


y_predict = model.predict(x_test)




#########catboost###########
#y_submit = model.predict(test_csv)[:,0]

y_submit = model.predict(test_csv)
y_submit = le.inverse_transform(y_submit)



acc = model.score(x_test, y_test)
print(acc)
import datetime
dt = datetime.datetime.now()
submission_csv['NObeyesdad'] = y_submit

#print(np.unique(y_submit, return_counts= True))
submission_csv.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_acc_{acc:4}.csv",index=False)