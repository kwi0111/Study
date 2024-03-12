import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
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

# x와 y를 분리
x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

y = np.reshape(y, (-1,1)) 

ohe = OneHotEncoder(sparse = False)
ohe.fit(y)
y_ohe = ohe.transform(y)

x_train, x_test, y_train, y_test = train_test_split(
                                                    x,
                                                    y_ohe,             
                                                    train_size=0.86,
                                                    random_state=2024,
                                                    stratify=y,
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

#2. 모델 구성 
model = Sequential()
model.add(Dense(10, input_dim=13, activation='swish'))
model.add(Dense(80, activation='swish'))
model.add(Dropout(0.2))
model.add(Dense(60, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(5, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(5, activation='swish'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))
# DESKTOP-0JLN3B0\AIA

#3.컴파일, 훈련
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import time as tm

rlr = ReduceLROnPlateau(monitor='val_loss',
                        patience=10,
                        mode='auto',
                        verbose=1,
                        factor=0.5,  # 반으로 줄여라.
                        )

learning_rate = 1
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=learning_rate))

es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=20,
                verbose=1,
                restore_best_weights=True
                )

start_time = tm.time()
model.fit(x_train, y_train, epochs=200, batch_size = 2024,
                validation_split=0.18,
                callbacks=[es,rlr],
                verbose=1
                )
end_time = tm.time()
run_time = round(end_time - start_time, 2)

#4.평가, 예측
results = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis = 1)
y_predict = np.argmax(y_predict, axis =1)
y_submit = np.argmax(y_submit, axis=1)
y_submit = le.inverse_transform(y_submit)

submission_csv['대출등급'] = y_submit
# submission_csv.to_csv(path + "submission_0312_1.csv", index=False)
# https://dacon.io/competitions/official/236214/mysubmission
acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average = 'macro') # [None, 'micro', 'macro', 'weighted'] 중에 하나

print('run time', run_time)
print("lr : {0}, ACC : {1}, loss : {2}, f1 : {3}".format(learning_rate, acc, results, f1))

'''
로스 :  0.2747177481651306
acc :  0.9085447192192078
f1 :  0.8775608046881204
'''

'''
lr : 0.0001, 로스 :0.26796308159828186 
lr : 0.0001, ACC : 0.5972407654650645

lr : 0.001, 로스 :0.12351109832525253 
lr : 0.001, ACC : 0.8396380358997182

lr : 0.01, 로스 :0.1271718442440033 
lr : 0.01, ACC : 0.8245067497403946

lr : 0.1, 로스 :0.35349321365356445 
lr : 0.1, ACC : 0.2868268802848242

lr : 1, 로스 :0.3531113564968109 
lr : 1, ACC : 0.2992879394748554
'''



'''rlr 적용
lr : 0.001, ACC : 0.8483162735499185, loss : 0.46867141127586365, f1 : 0.7012500283643066
lr : 0.01, ACC : 0.8535083815457647, loss : 0.41201385855674744, f1 : 0.7664473797644026
lr : 0.1, ACC : 0.7203678979379914, loss : 0.7039151191711426, f1 : 0.5660658325856368
'''
