# https://dacon.io/competitions/open/235610/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras. callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import random

#1.데이터
path = "c:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)    # [5497 rows x 13 columns]
print(train_csv.shape)  # (5497, 13)
print(train_csv.head) 

test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
print(test_csv)     
print(test_csv.shape)  # (1000, 12)
print(test_csv.info()) 

submission_csv = pd.read_csv(path + 'sample_submission.csv')
print(submission_csv)
print(submission_csv.shape)  # (1000, 2)
# 결측치 처리 
print(train_csv.isna().sum()) # 없다
# print(np.unique(x, return_counts=True)) 
train_csv['type'] = train_csv['type'].map({"white":1, "red":0}).astype(int)
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0}).astype(int)
print(train_csv)
print(test_csv)     # [1000 rows x 12 columns]
# x와 y를 분리
x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']
print(x.shape, y.shape)     # (5497, 12) (5497,)
print(np.unique(y, return_counts=True)) # (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))
print(pd.value_counts(y))
'''
6    2416
5    1788
7     924
4     186
8     152
3      26
9       5
'''
# 하이
# 원핫. 판다스
y_ohe = pd.get_dummies(y, dtype='int')
print(y_ohe)
print(y_ohe.shape)  # (5497, 7)
print(test_csv)
# r = random.randrange(1, 777)
x_train, x_test, y_train, y_test = train_test_split(
x, y_ohe,             
train_size=0.7,
random_state=123,
stratify=y_ohe,  
shuffle=True,
)
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
scaler = MinMaxScaler() # 클래스 정의
# scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)



#2. 모델 구성 
model = Sequential()
model.add(Dense(200, input_dim=12))
model.add(Dense(150))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(7, activation='softmax'))


#3.컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# es = EarlyStopping(monitor='val_loss',
#                 mode='min',
#                 patience=50,
#                 verbose=1,
#                 restore_best_weights=True
#                 )
mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath='../_data/_save/MCP/keras26_dacon_wine_MCP1.hdf5'
                      )
import time
start_time = time.time()   #현재 시간
model.fit(x_train, y_train, epochs=1000, batch_size=60,
        validation_split=0.2,
        callbacks=[mcp]
        )
end_time = time.time()   #끝나는 시간
#4.평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict)
y_submit = np.argmax(model.predict(test_csv), axis=1)      
print(y_submit)
submission_csv['quality'] = y_submit

print("로스 : ", results[0])  
print("acc : ", results[1])  

print("걸린시간 : ", round(end_time - start_time, 2),"초")