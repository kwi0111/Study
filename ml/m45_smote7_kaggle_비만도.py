import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score
import warnings
warnings.filterwarnings('ignore')


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

train_csv['NObeyesdad'] = train_csv['NObeyesdad'].str.replace('Insufficient_Weight', '0')
train_csv['NObeyesdad'] = train_csv['NObeyesdad'].str.replace('Normal_Weight', '1')
train_csv['NObeyesdad'] = train_csv['NObeyesdad'].str.replace('Overweight_Level_I', '2')
train_csv['NObeyesdad'] = train_csv['NObeyesdad'].str.replace('Overweight_Level_II', '3')
train_csv['NObeyesdad'] = train_csv['NObeyesdad'].str.replace('Obesity_Type_I', '4')
train_csv['NObeyesdad'] = train_csv['NObeyesdad'].str.replace('Obesity_Type_II', '5')
train_csv['NObeyesdad'] = train_csv['NObeyesdad'].str.replace('Obesity_Type_III', '6')
train_csv['NObeyesdad'] = train_csv['NObeyesdad'].str.replace('2I', '7')
train_csv['NObeyesdad'] = train_csv['NObeyesdad'].str.replace('4II', '8')
train_csv['NObeyesdad'] = train_csv['NObeyesdad'].str.replace('4I', '9')

# 데이터를 정수형으로 변환
train_csv['NObeyesdad'] = train_csv['NObeyesdad'].astype(int)

x = train_csv.drop(['NObeyesdad', 'SMOKE'], axis = 1)
y = train_csv['NObeyesdad']
test_csv = test_csv.drop(['SMOKE'], axis = 1)

############################################ 신뢰..? // 너무 많이 떨어져있는 아이들은 ACC보다 F1스코어로 봐야함
for i, v in enumerate(y):
    if v <= 4:  # 0, 1, 2, 3, 4 클래스를 0으로 변환
        y[i] = 0
    elif v == 5:  # 5 클래스를 1로 변환
        y[i] = 1
    elif v == 6:  # 6 클래스를 2로 변환
        y[i] = 2
    elif v == 7:  # 7 클래스를 3로 변환
        y[i] = 3
    else:  # 8, 9 클래스를 4로 변환
        y[i] = 4
print(pd.value_counts(y))

le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y , random_state=777, train_size=0.8,
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 3, # 트리 깊이
    'gamma' : 0,
    'min_child_weight' : 10,
    'min_child_weight' : 0,
    'subsample' : 0.4,
    'colsample_bytree' : 0.8,
    'colsample_bylevel' : 0.7,
    'colsample_bynode' : 1,
    'reg_alpha' : 0,
    'reg_lambda' : 1,
    'random_state' : 3377,
    'verbose' : 0,
}

########################## smote ############################### 데이터 증폭하는데 좋음
print("====================== smote 적용 =====================")
from imblearn.over_sampling import SMOTE
import sklearn as sk
print('사이킷런 : ', sk.__version__)    # 사이킷런 :  1.3.0

smote = SMOTE(random_state=123) # 랜덤 고정
x_train, y_train = smote.fit_resample(x_train, y_train) # 트레인 0.9 테스트 // 0.1은 그대로 (평가는 증폭 X)
print(pd.value_counts(y_train))

#2. 모델
model = XGBClassifier()
model.set_params(early_stopping_rounds=10, **parameters)

#.3 훈련
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=1,
          eval_metric='mlogloss',
          )

#4. 평가, 예측
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score : ', acc)
f1 = f1_score(y_test, y_predict, average='macro')
print('f1_score : ', f1)

'''
# acc_score :  0.8983622350674374
# f1_score :  0.8858111366140228
# ====================== smote만 적용 =====================
# 사이킷런 :  1.1.3
# 3    3255
# 4    3255
# 1    3255
# 2    3255
# 5    3255
# 0    3255
# 6    3255
# acc_score :  0.8966763005780347
# f1_score :  0.8841200466310835
====================== smote 적용 =====================
사이킷런 :  1.1.3
2    8746
0    8746
1    8746
acc_score :  0.9193159922928709
f1_score :  0.8881733250467124
'''







