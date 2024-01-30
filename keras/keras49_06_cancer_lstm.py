import numpy as np
from sklearn.datasets import load_breast_cancer     # 유방암
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터 
datasets = load_breast_cancer()
print(datasets.DESCR)   
print(datasets.feature_names)

x = datasets.data
y = datasets.target 
print(x.shape, y.shape)     # (569, 30) (569,)

print(np.unique(y, return_counts=True))       # (array([0, 1]), array([212, 357], dtype=int64)) ,라벨의 종류 별로 찾아줌 // 0과 1로 나눠져있다. // 0이 무엇인지 1이 무엇인지
print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212
print(pd.Series(y).value_counts())
# 1    357
# 0    212
print(pd.value_counts(y))                     # 행렬 데이터 일때 // mse로는 0과 1을 찾을수 없다. // 
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    random_state=123,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    )
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# scaler = MinMaxScaler() # 클래스 정의
scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train = x_train.reshape(-1, 15, 2)
x_test = x_test.reshape(-1, 15, 2)


#2. 모델 구성
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape = (15,2))) 
model.add(LSTM(30))    
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))    


#.3 컴파일 훈련 // 하나 포기하는게 과적합 안걸림 // 
model.compile(loss='binary_crossentropy',  # loss='binary_crossentropy' -> 이진 분류때 씀 (sigmoid) // 실제값과 예측값의 차이
              optimizer='adam',      
              metrics=['accuracy']         # 0인지 1인지를 맞추는 정확성 // acc = accuracy // 'mse', 'mae' 도 넣을수 있음
              )    
                # 둘중 하나는 무조건 'binary_crossentropy
es = EarlyStopping(monitor = 'val_loss',
                   mode='min',
                   patience=50,
                   verbose=1,
                   restore_best_weights=True
                   )
# mcp = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       verbose=1,
#                       save_best_only=True,
#                       filepath='../_data/_save/MCP/keras26_cancer_MCP1.hdf5'
#                       )
hist = model.fit(x_train, y_train, epochs=1000, batch_size=12, 
          validation_split=0.3, 
          callbacks=[es]        
          )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
y_predict = model.predict(x_test)
y_predict = y_predict.round()

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
r2 = r2_score(y_test, y_predict)    # (실제값, 예측값)
print(y_test)
print(y_predict)        # 시그모이드 때문에 정수값이 아니기 때문에 반올림 처리


def ACC(aaa, bbb):
    return (accuracy_score(aaa, bbb))
acc = ACC(y_test, y_predict)

print("ACC : " , acc)
print("로스, 정확도 : ", loss)
print("r2 스코어 : " , r2)  # r2 조금 못미더움 // 정확도는 predict에 대한 결과



# 씨피유 
# ACC :  0.9824561403508771
# 로스, 정확도 :  [1.6028051376342773, 0.9824561476707458]
# r2 스코어 :  0.9238222519211493
# 걸린시간 :  37.22 초

# ACC :  0.9649122807017544
# 로스, 정확도 :  [0.13235272467136383, 0.9649122953414917]
# r2 스코어 :  0.8476445038422986


'''
'''

