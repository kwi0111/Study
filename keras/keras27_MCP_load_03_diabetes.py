import numpy as np
import time
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split


#1. 데이터
datasets = load_diabetes()
x = datasets.data       # 학습해야 할 feed용 데이터
y = datasets.target     # label 데이터, 예측해야 할 (class) 데이터

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
scaler = MinMaxScaler() # 클래스 정의
# scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(300, input_dim = 10, activation='relu'))
# model.add(Dense(200, activation='relu')) 
# model.add(Dense(100, activation='relu')) 
# model.add(Dense(50, activation='relu')) 
# model.add(Dense(1))

# #3. 컴파일, 훈련
# model.compile(loss="mae", optimizer='adam')
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss',
#                    mode='min',
#                    patience=300,
#                    verbose=1,
#                    restore_best_weights=True
#                    )
# mcp = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       verbose=1,
#                       save_best_only=True,
#                       filepath='../_data/_save/MCP/keras26_diabetes_MCP1.hdf5'
#                       )
# model.fit(x_train, y_train, epochs=1000, batch_size=70, 
#           validation_split=0.2,
#           verbose=1,
#           callbacks=[es, mcp]
#           )   
model = load_model('../_data/_save/MCP/keras26_diabetes_MCP1.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                                           
y_predict = model.predict(x_test)
print(y_predict.shape)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)                                               

print("로스 : ", loss)
print("r2 스코어 : " , r2)





