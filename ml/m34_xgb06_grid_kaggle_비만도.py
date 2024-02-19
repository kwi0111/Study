import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from xgboost import XGBRFRegressor, XGBRFClassifier
import datetime
import time

# 데이터 로드
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


x = train_csv.drop('NObeyesdad', axis = 1)
y = train_csv['NObeyesdad']

le = LabelEncoder()
y = le.fit_transform(y)

# y = le.fit_transform(y)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.87, random_state=3, stratify=y)

# 데이터 스케일링
scaler = StandardScaler() 
# scaler = MaxAbsScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler() 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# XGBoost 모델 정의
random_state = np.random.randint(1, 100)
XGBoost_params = {
    "objective": "multi:softmax",  # 손실 함수를 설정합니다.
    "num_class": 7,  # 클래스의 수를 설정합니다.
    "verbosity": 0,  # 출력 메시지 레벨을 설정합니다.
    "seed": random_state,  # 랜덤 시드를 설정합니다.
    "learning_rate": 0.05,  # 학습률을 설정합니다.
    "n_estimators": 2000,  # 부스팅 라운드 수를 설정합니다.
    "colsample_bylevel": 0.8,  # 레벨별 컬럼 샘플링 비율을 설정합니다.
    "min_child_weight": 5,  # 리프 노드에 필요한 최소 가중치 합을 설정합니다.
    "booster": "gbtree",  # 부스팅 방법을 설정합니다.
    "subsample": 0.9,  # 데이터 샘플링 비율을 설정합니다.
    "max_depth": 10,  # 트리의 최대 깊이를 설정합니다.
    "gamma": 0,  # 리프 노드의 추가 분할을 위한 최소 손실 감소를 설정합니다.
    "reg_lambda": 7,  # L2 정규화 가중치를 설정합니다.
    "tree_method": "auto",  # 트리를 학습하기 위한 메소드를 설정합니다.
    "num_parallel_tree": 1,  # 각 반복에 사용할 병렬 트리의 수를 설정합니다.
    "nthread": -1  # 스레드 수를 설정합니다.
}


model = XGBRFClassifier(**XGBoost_params)

# 모델 학습
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

# 테스트 데이터 예측
y_pred = model.predict(x_test)
y_submit = model.predict(test_csv)

# 정확도 계산
accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy)

# submission_csv 업데이트
submission_csv['NObeyesdad'] = le.inverse_transform(y_submit)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))
print("걸린시간 :", round(end_time - start_time, 2), "초")

# 결과 저장
dt = datetime.datetime.now()
# submission_csv.to_csv(path + f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_acc_{accuracy:.4f}.csv", index=False)

# accuracy_score : 0.8947758429047795
# 걸린시간 : 11.66 초
