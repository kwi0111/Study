import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical #



#1. 데이터
path = "C:\\_data\\dacon\\dechul\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0 )
print(train_csv.shape)  # (96294, 14)
test_csv = pd.read_csv(path + "test.csv", index_col=0 )
print(test_csv.shape)  # (64197, 13)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv.shape)  # (64197, 2)

# train_csv[train_csv.iloc[:,:] == 'Unknown'] = np.NaN
# train_csv.isnull().sum() # 변수별 결측치의 갯수
# print(train_csv.isnull().sum())
# train_csv['근로기간'] = train_csv['근로기간'].fillna(train_csv['근로기간'].mean())     # train에서 없어진 결측치를 평균인 중간값으로 채움.
# print(train_csv.info())


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
x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']
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

# mms = MinMaxScaler()
# mms.fit(x)
# x = mms.transform(x)
# test_csv=mms.transform(test_csv)

y = np.reshape(y, (-1,1)) 
# y = np.array()


ohe = OneHotEncoder(sparse = False)
ohe.fit(y)
y_ohe = ohe.transform(y)
print(y.shape)  



# y_ohe = pd.get_dummies(y, dtype='int')
# print(y_ohe)   
# print(x.shape, y.shape)   # (96294, 13) (96294, 1) // (96294, ) 벡터 형태 -> reshape를 이용해 행렬로 바꿔줘야함


x_train, x_test, y_train, y_test = train_test_split(
                                                    x,
                                                    y_ohe,             
                                                    train_size=0.93,    #
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
# 스케일러 연속으로 쓸수있다. // 레이어도 정제 가능 //

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)



#2. 모델 구성 

model = Sequential()
model.add(Dense(10, input_dim=13, activation='swish'))
model.add(Dense(800, activation='swish')) # 80
model.add(Dense(600, activation='swish'))
model.add(Dense(20, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(5, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(5, activation='swish'))
model.add(Dense(800, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(7, activation='softmax'))

#2. 모델구성(함수형)
# input1 = Input(shape=(13, ))
# dense1 = Dense(10, activation='swish')(input1)
# dense2 = Dense(80, activation='swish')(dense1)
# dense3 = Dense(60, activation='swish')(dense2)
# dense4 = Dense(20, activation='swish')(dense3)
# dense5 = Dense(10, activation='swish')(dense4)
# dense6 = Dense(5, activation='swish')(dense5)
# dense7 = Dense(10, activation='swish')(dense6)
# dense8 = Dense(10, activation='swish')(dense7)
# dense9 = Dense(10, activation='swish')(dense8)
# dense10 = Dense(5, activation='swish')(dense9)
# dense11 = Dense(10, activation='swish')(dense10)
# output1 = Dense(7, activation='softmax')(dense11)
# model = Model(inputs = input1, outputs = output1)


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
                patience=2000,
                verbose=1,
                restore_best_weights=True
                )
mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath=filepath,
                      )

model.fit(x_train, y_train, epochs=15000, batch_size = 1424,
                validation_split=0.13,  #
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
f1 = f1_score(y_test, y_predict, average='macro')
acc = accuracy_score(y_test, y_predict)
print("로스 : ", results[0])  
print("acc : ", results[1])  
print("f1 : ", f1)  
submission_csv.to_csv(path + "submission_0117_2.csv", index=False)

'''
로스 :  0.16252931952476501
acc :  0.9449636340141296
f1 :  0.9268314268128603

발 0.2
로스 :  0.18973031640052795
acc :  0.9349688291549683
f1 :  0.9165771683299866

발 0.17
로스 :  0.187760591506958
acc :  0.9360072612762451
f1 :  0.9152796509835436

발 0.15
로스 :  0.18486541509628296
acc :  0.9378893971443176
f1 :  0.9223869852274916

발 0.13 0.93
로스 :  0.1594099998474121
acc :  0.9454179406166077
f1 :  0.9290122602294401

발 0.1
로스 :  0.2445290982723236
acc :  0.9186137318611145
f1 :  0.9042328663195657

발 0.12
로스 :  0.20494093000888824
acc :  0.9302959442138672
f1 :  0.9129390647928908

발 0.13 배치 1424 -> 1524
로스 :  0.18184959888458252
acc :  0.9397066235542297
f1 :  0.9155483973652124

배치 1524 트레인사이즈 0.93
로스 :  0.19462506473064423
acc :  0.930425763130188
f1 :  0.9146101936659571

배치 1024 발 0.17 트 0.93
로스 :  0.18713201582431793
acc :  0.9390298128128052
f1 :  0.9228856359027988

배치 1224 페이션 1500
로스 :  0.19295533001422882
acc :  0.9384364485740662
f1 :  0.9174710104515917

2024

'''



