# save_best_only 와 restore_best_weights에 대한 고찰

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_boston()        # 변수에 집어 넣은다음 프린트
x = datasets.data           # x에서 스케일링
y = datasets.target         # y 건들지 않는다.
print(x.shape, y.shape)  #(506, 13), (506,)
print(datasets.feature_names)
print(datasets.DESCR)   #설명하다. 묘사하다. 데이터셋의 내용

x_train, x_test, y_train, y_test = train_test_split(x, y,               
                                                    train_size=0.7,
                                                    random_state=1140,     
                                                    shuffle=True,
                                                    )

#2. 모델 구성 
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련 
from keras. callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()  
print(date) # 2024-01-17 10:55:11.015537
print(type(date))   # <class 'datetime.datetime'> 시간 데이터
date = date.strftime("%m%d_%H%M")   # "%m%d_%H%M" 월 일 시간 분 // _는 문자
print(date) # 0117_1059
print(type(date))   # <class 'str'> 문자열

path = '../_data/_save/MCP/'  # 문자열로 저장
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'  # 히스토리로 반환 되는 놈들 // 훈련 횟수 - 발로스 // 04d : 4자리수 까지  // 04f : 소수 4번째 자리 까지 // ex) 1000-0.3333.hdf5
filepath = "".join([path, 'k25_', date, '_', filename])  # ""은 더하기 개념 (path + date + filename)
# ../_data/_save/MCP/k25_0117_1059_0001-0.3333.hdf5


es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=10,
                   verbose=1,
                   restore_best_weights=True,
                   )
mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath=filepath,  # 저장1
                      )
model.compile(loss="mse", optimizer='adam')
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[es, mcp, ], validation_split=0.2) # 핏 다음에 가중치


# model = load_model('../_data/_save/MCP/keras25_MCP1.hdf5')

#4. 평가, 예측
print("--------------- 1. 기본 출력 ---------------------")
loss = model.evaluate(x_test, y_test, verbose=0)
y_predict = model.predict(x_test, verbose=0)

r2 = r2_score(y_test, y_predict)
print("로스 : ", loss)
print("r2 스코어 : " , r2)

print("------------------------------------")
print(hist.history['val_loss'])
print("------------------------------------")

# #restore_best_weights
# save_best_only 

# True, True    제일 좋은거 남음 // 그냥 이게 낫다.
# True, False   모든 에포 별로 저장됨
# False, True   제일 좋은거 남음
# False, False  다 저장


