import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, LSTM
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split


#1. 데이터
datasets = load_diabetes()
x = datasets.data       
y = datasets.target     

print(x)
print(y)
print(x.shape, y.shape) # (442, 10) (442,)
print(datasets.feature_names)   # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.88,
                                                    random_state=123,
                                                    shuffle=True
                                                    )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# scaler = MinMaxScaler() # 클래스 정의
scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train = x_train.reshape(-1, 10, 1)
x_test = x_test.reshape(-1, 10, 1)

#2. 모델 구성
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape = (10,1))) 
model.add(LSTM(30))    
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(1))        

#3. 컴파일, 훈련
model.compile(loss="mae", optimizer='adam')
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=300,
                   verbose=1,
                   restore_best_weights=True
                   )
model.fit(x_train, y_train, epochs=1000, batch_size=70, 
          validation_split=0.2,
          verbose=1,
          callbacks=[es]
          )   
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                                           
y_predict = model.predict(x_test)
print(y_predict.shape)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)                                               

print("로스 : ", loss)
print("r2 스코어 : " , r2)

# 씨피유
# 로스 :  44.54854202270508
# r2 스코어 :  0.4272444001845447
# 걸린시간 :  21.96 초

# CNN
# 로스 :  34.18056106567383
# r2 스코어 :  0.7001776277092553
# 걸린시간 :  14.89 초


# LSTM
# 로스 :  45.068016052246094
# r2 스코어 :  0.5100224680285621
