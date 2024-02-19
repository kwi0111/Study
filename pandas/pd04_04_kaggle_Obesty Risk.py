from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

path = 'C:\\_data\\kaggle\\Obesity_Risk\\'
train_csv = pd.read_csv(path+"train.csv", index_col=0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv", index_col=0)


print(train_csv.shape, test_csv.shape, submission_csv.shape)    # (20758, 17) (13840, 16) (13840, 2)
print(train_csv.columns)

for label in train_csv:
        print(train_csv[label].isna().sum())    # 결측치 없음을 확인

class_label = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS','NObeyesdad']


# train csv 라벨들 확인
for label in train_csv:
        if label in class_label:
                print(label ,np.unique(train_csv[label], return_counts=True))
# Gender (array(['Female', 'Male'], dtype=object), array([10422, 10336], dtype=int64))
# family_history_with_overweight (array(['no', 'yes'], dtype=object), array([ 3744, 17014], dtype=int64))
# FAVC (array(['no', 'yes'], dtype=object), array([ 1776, 18982], dtype=int64))
# CAEC (array(['Always', 'Frequently', 'Sometimes', 'no'], dtype=object), array([  478,  2472, 17529,   279], dtype=int64))
# SMOKE (array(['no', 'yes'], dtype=object), array([20513,   245], dtype=int64))
# SCC (array(['no', 'yes'], dtype=object), array([20071,   687], dtype=int64))
# CALC (array(['Frequently', 'Sometimes', 'no'], dtype=object), array([  529, 15066,  5163], dtype=int64))
# MTRANS (array(['Automobile', 'Bike', 'Motorbike', 'Public_Transportation',
#        'Walking'], dtype=object), array([ 3534,    32,    38, 16687,   467], dtype=int64))
# NObeyesdad (array(['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
#        'Obesity_Type_II', 'Obesity_Type_III', 'Overweight_Level_I',
#        'Overweight_Level_II'], dtype=object), array([2523, 3082, 2910, 3248, 4046, 2427, 2522], dtype=int64))


 # test csv 라벨들 확인
for label in test_csv:
        if label in class_label:
                print(label, np.unique(test_csv[label], return_counts=True))
# Gender (array(['Female', 'Male'], dtype=object), array([6965, 6875], dtype=int64))
# family_history_with_overweight (array(['no', 'yes'], dtype=object), array([ 2456, 11384], dtype=int64))
# FAVC (array(['no', 'yes'], dtype=object), array([ 1257, 12583], dtype=int64))
# CAEC (array(['Always', 'Frequently', 'Sometimes', 'no'], dtype=object), array([  359,  1617, 11689,   175], dtype=int64))
# SMOKE (array(['no', 'yes'], dtype=object), array([13660,   180], dtype=int64))
# SCC (array(['no', 'yes'], dtype=object), array([13376,   464], dtype=int64))
# CALC (array(['Always', 'Frequently', 'Sometimes', 'no'], dtype=object), array([   2,  346, 9979, 3513], dtype=int64))
# MTRANS (array(['Automobile', 'Bike', 'Motorbike', 'Public_Transportation',
#        'Walking'], dtype=object), array([ 2405,    25,    19, 11111,   280], dtype=int64))

test_csv.loc[test_csv['CALC'] == 'Always', 'CALC'] = 'Frequently'

print(train_csv.head)
 # train head 확인
# <bound method NDFrame.head of        
#         Gender      Age    Height      Weight   family_history_with_overweight FAVC      FCVC     NCP        CAEC   SMOKE    CH2O   SCC    FAF       TUE       CALC                 MTRANS           NObeyesdad
# id
# 0        Male  24.443011  1.699998   81.669950                            yes  yes  2.000000  2.983297   Sometimes    no  2.763573  no  0.000000  0.976473  Sometimes  Public_Transportation  Overweight_Level_II
# 1      Female  18.000000  1.560000   57.000000                            yes  yes  2.000000  3.000000  Frequently    no  2.000000  no  1.000000  1.000000         no             Automobile        Normal_Weight
# 2      Female  18.000000  1.711460   50.165754                            yes  yes  1.880534  1.411685   Sometimes    no  1.910378  no  0.866045  1.673584         no  Public_Transportation  Insufficient_Weight
# 3      Female  20.952737  1.710730  131.274851                            yes  yes  3.000000  3.000000   Sometimes    no  1.674061  no  1.467863  0.780199  Sometimes  Public_Transportation     Obesity_Type_III
# 4        Male  31.641081  1.914186   93.798055                            yes  yes  2.679664  1.971472   Sometimes    no  1.979848  no  1.967973  0.931721  Sometimes  Public_Transportation  Overweight_Level_II
# ...       ...        ...       ...         ...                            ...  ...       ...       ...         ...   ...       ...  ..       ...       ...        ...                    ...                  ...
# 20753    Male  25.137087  1.766626  114.187096                            yes  yes  2.919584  3.000000   Sometimes    no  2.151809  no  1.330519  0.196680  Sometimes  Public_Transportation      Obesity_Type_II
# 20754    Male  18.000000  1.710000   50.000000                             no  yes  3.000000  4.000000  Frequently    no  1.000000  no  2.000000  1.000000  Sometimes  Public_Transportation  Insufficient_Weight
# 20755    Male  20.101026  1.819557  105.580491                            yes  yes  2.407817  3.000000   Sometimes    no  2.000000  no  1.158040  1.198439         no  Public_Transportation      Obesity_Type_II
# 20756    Male  33.852953  1.700000   83.520113                            yes  yes  2.671238  1.971472   Sometimes    no  2.144838  no  0.000000  0.973834         no             Automobile  Overweight_Level_II
# 20757    Male  26.680376  1.816547  118.134898                            yes  yes  3.000000  3.000000   Sometimes    no  2.003563  no  0.684487  0.713823  Sometimes  Public_Transportation      Obesity_Type_II

print(train_csv.columns)
'''
x_labelEncoder = LabelEncoder()
train_csv['Gender'] = x_labelEncoder.fit_transform(train_csv['Gender'])
train_csv['family_history_with_overweight'] = x_labelEncoder.fit_transform(train_csv['family_history_with_overweight'])
train_csv['FAVC'] = x_labelEncoder.fit_transform(train_csv['FAVC'])
train_csv['CAEC'] = x_labelEncoder.fit_transform(train_csv['CAEC'])
train_csv['SMOKE'] = x_labelEncoder.fit_transform(train_csv['SMOKE'])
train_csv['SCC'] = x_labelEncoder.fit_transform(train_csv['SCC'])
train_csv['CALC'] = x_labelEncoder.fit_transform(train_csv['CALC'])
train_csv['MTRANS'] = x_labelEncoder.fit_transform(train_csv['MTRANS'])

x_labelEncoder = LabelEncoder()
test_csv['Gender'] = x_labelEncoder.fit_transform(test_csv['Gender'])
test_csv['family_history_with_overweight'] = x_labelEncoder.fit_transform(test_csv['family_history_with_overweight'])
test_csv['FAVC'] = x_labelEncoder.fit_transform(test_csv['FAVC'])
test_csv['CAEC'] = x_labelEncoder.fit_transform(test_csv['CAEC'])
test_csv['SMOKE'] = x_labelEncoder.fit_transform(test_csv['SMOKE'])
test_csv['SCC'] = x_labelEncoder.fit_transform(test_csv['SCC'])
test_csv['CALC'] = x_labelEncoder.fit_transform(test_csv['CALC'])
test_csv['MTRANS'] = x_labelEncoder.fit_transform(test_csv['MTRANS'])

y_labelEncoder = LabelEncoder()
train_csv['NObeyesdad'] = y_labelEncoder.fit_transform(train_csv['NObeyesdad'])
# print(train_csv.head)
 # 라벨 인코딩 후 train.head 확인
<bound method NDFrame.head of        
       Gender     Age      Height     Weight   family_history_with_overweight    FAVC    CVC       NCP     CAEC  SMOKE   CH2O    SCC    FAF       TUE      CALC   MTRANS    NObeyesdad
id
0           1  24.443011  1.699998   81.669950                               1     1  2.000000  2.983297     2      0  2.763573    0  0.000000  0.976473     1       3           6
1           0  18.000000  1.560000   57.000000                               1     1  2.000000  3.000000     1      0  2.000000    0  1.000000  1.000000     2       0           1
2           0  18.000000  1.711460   50.165754                               1     1  1.880534  1.411685     2      0  1.910378    0  0.866045  1.673584     2       3           0
3           0  20.952737  1.710730  131.274851                               1     1  3.000000  3.000000     2      0  1.674061    0  1.467863  0.780199     1       3           4
4           1  31.641081  1.914186   93.798055                               1     1  2.679664  1.971472     2      0  1.979848    0  1.967973  0.931721     1       3           6
...       ...        ...       ...         ...                             ...   ...       ...       ...   ...    ...       ...  ...       ...       ...   ...     ...         ...
20753       1  25.137087  1.766626  114.187096                               1     1  2.919584  3.000000     2      0  2.151809    0  1.330519  0.196680     1       3           3
20754       1  18.000000  1.710000   50.000000                               0     1  3.000000  4.000000     1      0  1.000000    0  2.000000  1.000000     1       3           0
20755       1  20.101026  1.819557  105.580491                               1     1  2.407817  3.000000     2      0  2.000000    0  1.158040  1.198439     2       3           3
20756       1  33.852953  1.700000   83.520113                               1     1  2.671238  1.971472     2      0  2.144838    0  0.000000  0.973834     2       0           6
20757       1  26.680376  1.816547  118.134898                               1     1  3.000000  3.000000     2      0  2.003563    0  0.684487  0.713823     1       3           3



 # P 검정
import scipy.stats as stats
for label in train_csv:
    print(label," ",stats.pearsonr(train_csv['NObeyesdad'],train_csv[label]))
Gender   PearsonRResult(statistic=0.046574912033978184, pvalue=1.8990220650683642e-11)
Age   PearsonRResult(statistic=0.2830183712239907, pvalue=0.0)
Height   PearsonRResult(statistic=0.060785550400480136, pvalue=1.8621467518362792e-18)
Weight   PearsonRResult(statistic=0.43182097207728903, pvalue=0.0)
family_history_with_overweight   PearsonRResult(statistic=0.32132484319938587, pvalue=0.0)
        FAVC   PearsonRResult(statistic=0.010176246176913503, pvalue=0.14261934009881228)
FCVC   PearsonRResult(statistic=0.0410763864808035, pvalue=3.2139644150763383e-09)
NCP   PearsonRResult(statistic=-0.09115416942846906, pvalue=1.4953264119959517e-39)
CAEC   PearsonRResult(statistic=0.297419757072015, pvalue=0.0)
        SMOKE   PearsonRResult(statistic=-0.0013927633524938529, pvalue=0.84097046043243)
CH2O   PearsonRResult(statistic=0.18709958001025073, pvalue=7.344610346043163e-163)
SCC   PearsonRResult(statistic=-0.06517135385355988, pvalue=5.504265380370881e-21)
FAF   PearsonRResult(statistic=-0.09664292513984239, pvalue=2.9003365419591887e-44)
TUE   PearsonRResult(statistic=-0.07603955730067295, pvalue=5.284478962295593e-28)
CALC   PearsonRResult(statistic=-0.16849742485140495, pvalue=5.0277297570113e-132)
MTRANS   PearsonRResult(statistic=-0.07743006081693204, pvalue=5.594245339542871e-29)
NObeyesdad   PearsonRResult(statistic=1.0, pvalue=0.0) 

 BMI 컬럼 추가
train_csv['BMI'] = train_csv['Weight'] / (train_csv['Height']*train_csv['Height'])
test_csv['BMI'] = test_csv['Weight'] / (test_csv['Height']*test_csv['Height'])

이상치 제거 
age_q1 = train_csv['Age'].quantile(0.25)
age_q3 = train_csv['Age'].quantile(0.75)
age_gap = (age_q3 - age_q1 ) * 1.5
age_under = age_q1 - age_gap
age_upper = age_q3 + age_gap
train_csv = train_csv[train_csv['Age']>=age_under]
train_csv = train_csv[train_csv['Age']<=age_upper]

weight_q1 = train_csv['Weight'].quantile(0.25)
weight_q3 = train_csv['Weight'].quantile(0.75)
weight_gap = (weight_q3 - weight_q1 ) * 1.5
weight_under = weight_q1 - weight_gap
weight_upper = weight_q3 + weight_gap
train_csv = train_csv[train_csv['Weight']>=weight_under]
train_csv = train_csv[train_csv['Weight']<=weight_upper]

x = train_csv.drop(['NObeyesdad'], axis=1) # P검정에 의거하여 FAVC와 SMOKE 제거
y = train_csv['NObeyesdad']

# 최대 최소 1분위 3분위 구하기
for label in x:         
        if label in class_label:
                continue
        print(f"{label:30}: max={max(x[label]):<10}  min={min(x[label]):<10}  q1={x[label].quantile(0.25):<10}  q3={x[label].quantile(0.75):<10}")
Age                           : max=61.0        min=14.0        q1=20.0        q3=26.0
Height                        : max=1.975663    min=1.45        q1=1.631856    q3=1.762887
Weight                        : max=165.057269  min=39.0        q1=66.0        q3=111.600553
FCVC                          : max=3.0         min=1.0         q1=2.0         q3=3.0
NCP                           : max=4.0         min=1.0         q1=3.0         q3=3.0
CH2O                          : max=3.0         min=1.0         q1=1.792022    q3=2.549617
FAF                           : max=3.0         min=0.0         q1=0.008013    q3=1.587406
TUE                           : max=2.0         min=0.0         q1=0.0         q3=1.0


scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
x = scaler.fit_transform(x)
test_csv = scaler.fit_transform(test_csv)

# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = StandardScaler()
# scaler = RobustScaler()
x = scaler.fit_transform(x)
test_csv = scaler.fit_transform(test_csv)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333, stratify=y)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, RandomizedSearchCV
# import optuna

param = {'iterations': [900], 'depth': [4], 'learning_rate': [0.06], 'task_type' : ['GPU']}
# param = {'iterations': [500,900, 1500, 2500], 'depth': [4], 'learning_rate': [0.06], 'task_type' : ['GPU']}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=333)

model = GridSearchCV(CatBoostClassifier(), param, cv=kfold, verbose=1, refit=False, n_jobs=-1)
model.fit(x_train,y_train)

y_pre_best = model.best_estimator_.predict(x_test)
acc = accuracy_score(y_test, y_pre_best)

y_submit = model.best_estimator_.predict(test_csv)
y_submit = y_labelEncoder.inverse_transform(y_submit)

print("최적의 매개변수: ", model.best_estimator_)
print("최적의 파라미터: ", model.best_params_)
print(param)
print("ACC: ",acc) 
def objectiveCAT(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'thread_count': 4,
        'verbose': False
    }

    model = CatBoostClassifier(**params)

    # Train the model
    model.fit(x_train, y_train, verbose=False)

    # Make predictions on the validation set
    val_preds = model.predict(x_test)

    # Calculate accuracy on the validation set
    accuracy = accuracy_score(y_test, val_preds)

    return accuracy

def objectiveLGBM(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': -1,
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 40),
    }

    model = LGBMClassifier(**params)

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions on the validation set
    val_preds = model.predict(x_test)

    # Calculate accuracy on the validation set
    accuracy = accuracy_score(y_test, val_preds)

    return accuracy

def objectiveXGB(trial):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'max_depth' : trial.suggest_int('max_depth', 8, 16),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
        'gamma' : trial.suggest_int('gamma', 1, 3),
        'learning_rate' : 0.01,
        'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        'nthread' : -1,
        # 'tree_method' : 'gpu_hist',
        # 'predictor' : 'gpu_predictor',
        'lambda' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha' : trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample' : trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
        'random_state' : 1127
    }
    
    # 학습 모델 생성
    model = XGBClassifier(**param)
    xgb_model = model.fit(x_train, y_train, verbose=True) # 학습 진행
    
    # 모델 성능 확인
    score = accuracy_score(xgb_model.predict(x_test), y_test)
    
    return score

# study = optuna.create_study(direction='maximize')
# study.optimize(objectiveLGBM, n_trials=100)

# best_params = study.best_params
# print(best_params)

# optuna.visualization.plot_param_importances(study)      # 파라미터 중요도 확인 그래프
# optuna.visualization.plot_optimization_history(study)   # 최적화 과정 시각화

# params = {'iterations': 452, 'learning_rate': 0.18947052287744456, 'depth': 6, 'l2_leaf_reg': 6.8398928223584035, 'border_count': 243} # catboost, always 처리 안함
params = {'iterations': 777, 'learning_rate': 0.10152509183335467, 'depth': 6, 'l2_leaf_reg': 1.4112760375644173, 'border_count': 154} # catboost, always 처리
# params = {'n_estimators': 2391, 'max_depth': 16, 'min_child_weight': 19, 'gamma': 1, 'colsample_bytree': 0.8, 'lambda': 2.7858366632566747, 'alpha': 0.004919261757405025, 'subsample': 0.8}    #xgboost, always 처리
model = CatBoostClassifier(**params)
# model = XGBClassifier(**params)

# Train the model
model.fit(x_train, y_train, verbose=False)

acc = model.score(x_test,y_test)

# Make predictions on the validation set
y_submit = model.predict(test_csv)
y_submit = y_labelEncoder.inverse_transform(y_submit)

# 최적의 매개변수:  <catboost.core.CatBoostClassifier object at 0x000001FEBBDC3EE0>
# 최적의 파라미터:  {'depth': 4, 'iterations': 1000, 'learning_rate': 0.07}
# {'iterations': [1000, 1500, 2000], 'depth': [4, 5, 6], 'learning_rate': [0.01, 0.05, 0.07, 0.1]}
# ACC:  0.9065510597302505

from sklearn.utils import all_estimators
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pandas as pd
import datetime

# param = {'iterations': 1000, 'depth': 4, 'learning_rate': 0.07}
# model = CatBoostClassifier(**param)
# model.fit(x_train,y_train)

# acc = model.score(x_test,y_test)
# y_submit = model.predict(test_csv)
# y_submit = y_labelEncoder.inverse_transform(y_submit)

submit_csv = pd.read_csv(path+"sample_submission.csv")
print(submit_csv.columns)   # ['id', 'NObeyesdad']
submit_csv['NObeyesdad'] = y_submit

dt = datetime.datetime.now()
# submit_csv.to_csv(path+f"submit/{dt.day}acc_{acc:.6f}.csv",index=False)

# print(param)
print("ACC: ",acc)



# n_estimators=1000, learning_rate=0.2, max_depth=4, random_state=32
# ACC:  0.9041425818882466

# n_estimators=1000, learning_rate=0.15, max_depth=4, random_state=32
# ACC:  0.9048651252408478

# {'iterations': 1000, 'depth': 5, 'learning_rate': 0.1}
# ACC:  0.9111271676300579

# {'iterations': 1500, 'depth': 5, 'learning_rate': 0.1}
# ACC:  0.9123314065510597

# 열 제거 안함
# 최적의 파라미터:  {'depth': 4, 'iterations': 900, 'learning_rate': 0.06, 'task_type': 'GPU'}
# {'iterations': [900], 'depth': [4], 'learning_rate': [0.06], 'task_type': ['GPU']}
# ACC:  0.9132947976878613
# Index(['id', 'NObeyesdad'], dtype='object')
# ACC:  0.9132947976878613
# 실제 ACC: 0.8945

# FAVC만 제거
# 최적의 파라미터:  {'depth': 4, 'iterations': 900, 'learning_rate': 0.06, 'task_type': 'GPU'}
# {'iterations': [900], 'depth': [4], 'learning_rate': [0.06], 'task_type': ['GPU']}
# ACC:  0.9096820809248555
# Index(['id', 'NObeyesdad'], dtype='object')
# ACC:  0.9096820809248555
# 실제 ACC: 0.444

# SMOKE만 제거
# 최적의 파라미터:  {'depth': 4, 'iterations': 900, 'learning_rate': 0.06, 'task_type': 'GPU'}
# {'iterations': [900], 'depth': [4], 'learning_rate': [0.06], 'task_type': ['GPU']}
# ACC:  0.9135356454720617
# Index(['id', 'NObeyesdad'], dtype='object')
# ACC:  0.9135356454720617

'''
