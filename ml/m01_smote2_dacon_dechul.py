# 스모트 적용

import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import time


#1. 데이터
path = "C:\\_data\\dacon\\dechul\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0 )
print(train_csv.shape)  # (96294, 14)
test_csv = pd.read_csv(path + "test.csv", index_col=0 )
print(test_csv.shape)  # (64197, 13)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv.shape)  # (64197, 2)


# 라벨 엔코더
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder() # 대출기간, 대출목적, 근로기간, 주택소유상태 // 라벨 인코더 : 카테고리형 피처를 숫자형으로 변환
train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = le.fit_transform(train_csv['대출목적'])
train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])

test_csv['주택소유상태'] = le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = le.fit_transform(test_csv['대출목적'])
test_csv['대출기간'] = test_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
test_csv['근로기간'] = le.fit_transform(test_csv['근로기간'])

train_csv['대출등급'] = le.fit_transform(train_csv['대출등급']) # 마지막에 와야함

print(train_csv.describe)
print(test_csv.describe)

print(train_csv.shape)
print(test_csv.shape)
print(train_csv.dtypes)
print(test_csv.dtypes)

# x와 y를 분리
x = train_csv.drop(['대출등급','총계좌수'], axis=1)
y = train_csv['대출등급']
test_csv = test_csv.drop(['총계좌수'], axis=1)
print(x.shape, y.shape) # (96294, 13) (96294,)
print(pd.value_counts(y))
# 대출등급
# 1    28817
# 2    27623
# 0    16772
# 3    13354
# 4     7354
# 5     1954
# 6      420
print(np.unique(y, return_counts=True))   # (array([0, 1, 2, 3, 4, 5, 6]), array([16772, 28817, 27623, 13354,  7354,  1954,   420],

print(y.shape)  # (96294,)


y = np.reshape(y, (-1,1)) 
# y = np.array()

ohe = OneHotEncoder(sparse = False)
ohe.fit(y)
y_ohe = ohe.transform(y)
# print(y.shape)  # (96294, 1)


# y_ohe = pd.get_dummies(y, dtype='int')
# print(y_ohe)   
# print(x.shape, y.shape)   # (96294, 13) (96294, 1) // (96294, ) 벡터 형태 -> reshape를 이용해 행렬로 바꿔줘야함
print(x.shape, y.shape) # (96294, 12) (96294, 1)

x_train, x_test, y_train, y_test = train_test_split(
                                                    x,
                                                    y_ohe,             
                                                    train_size=0.95,
                                                    random_state=2024,
                                                    stratify=y_ohe,
                                                    shuffle=True,
                                                    )
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# scaler = MinMaxScaler() # 클래스 정의
scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# print(y_test)
# print(y_train)
# ########################## smote ############################### 데이터 증폭하는데 좋음
print("====================== smote 적용 =====================")
print(np.unique(y_train, return_counts=True))   # (array([0., 1.]), array([514206,  85701], dtype=int64))
print(np.unique(y_test, return_counts=True))   # (array([0., 1.]), array([63558, 10593], dtype=int64))

start_time = time.time()
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=2024) # 랜덤 고정
x_train, y_train = smote.fit_resample(x_train, y_train)
print(x_train.shape, y_train.shape) # (179529, 12) (179529, 7)
end_time = time.time()
print(np.unique(y_train, return_counts=True))   # (array([0, 1]), array([1077174,  179529], dtype=int64))
print(np.unique(y_test, return_counts=True))   # (array([0., 1.]), array([63558, 10593], dtype=int64))


#2. 모델 구성 

model = Sequential()
model.add(Dense(10, input_dim=12, activation='swish'))
model.add(Dense(500, activation='swish'))
model.add(Dropout(0.1))
model.add(Dense(580, activation='swish'))
model.add(Dropout(0.1))
model.add(Dense(280, activation='swish'))
model.add(Dense(180, activation='swish'))
model.add(Dense(80, activation='swish'))
model.add(Dense(50, activation='swish'))
model.add(Dense(50, activation='swish'))
model.add(Dense(7, activation='softmax'))


#3.컴파일, 훈련
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
MCP_path = "../_data/_save/MCP/"
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([MCP_path, 'k23_', date, '_', filename])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=300,
                verbose=1,
                restore_best_weights=True
                )
mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath=filepath,
                      )

model.fit(x_train, y_train, epochs=15000, batch_size = 1324,
                validation_split=0.18,  
                callbacks=[es, mcp],
                verbose=1
                )

#4.평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
arg_pre = np.argmax(y_predict, axis=1)    #  argmax : NumPy 배열에서 가장 높은 값을 가진 값의 인덱스를 반환
arg_test = np.argmax(y_test, axis=1)
y_submit = model.predict(test_csv)
submit = np.argmax(y_submit, axis=1)
submitssion = le.inverse_transform(submit)
      
submission_csv['대출등급'] = submitssion
y_predict = ohe.inverse_transform(y_predict)
y_test = ohe.inverse_transform(y_test)
f1 = f1_score(arg_test, arg_pre, average='macro')
print("로스 : ", results[0])  
print("acc : ", results[1])  
print("f1 : ", f1)  
submission_csv.to_csv(path + "submission_0126_1.csv", index=False)

print("걸린시간 : ", round(end_time - start_time, 2),"초")

'''
'''



'''
로스 :  0.16252931952476501
acc :  0.9449636340141296
f1 :  0.9268314268128603

증폭
로스 :  0.4461914002895355
acc :  0.8480128645896912
f1 :  0.7989282376487419

로스 :  0.3726523220539093
acc :  0.8718587756156921
f1 :  0.8229297107796958
걸린시간 :  5.3 초


'''



