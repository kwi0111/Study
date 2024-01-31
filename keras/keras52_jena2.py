# 5일분(720행) 을 훈련시켜서 하루(144행) 뒤를 예측

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#1.데이터
path = "c://_data//kaggle//jena//"

dataset = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

# size = 720
# def split_xy(dataset, size, y_col):
#     result_x = []
#     result_y = []
#     for i in range(len(dataset) - size + 1):
#         result_x.append(dataset[i:i+size])
#         y_row = dataset.iloc[i+size]
#         result_y.append(y_row[y_col])
#     return np.array(result_x), np.array(result_y)

# x, y = split_xy(dataset, size, 'T (degC)')

# print(x)
# print(y)
# print(x.shape, y.shape)

# data RNN에 맞게 변환
def split_xy(data, time_step, y_col,y_gap=0):
    result_x = []
    result_y = []
    
    num = len(data) - (time_step+y_gap)                 # x만자른다면 len(data)-time_step+1이지만 y도 잘라줘야하므로 +1이 없어야함
    for i in range(num):
        result_x.append(data[i : i+time_step])  # i 부터  time_step 개수 만큼 잘라서 result_x에 추가
        y_row = data.iloc[i+time_step+y_gap]          # i+time_step번째 행, 즉 result_x에 추가해준 바로 다음순번 행
        result_y.append(y_row[y_col])           # i+time_step번째 행에서 원하는 열의 값만 result_y에 추가
    
    return np.array(result_x), np.array(result_y)

TRAIN_SIZE = 720
PREDICT_GAP = 144
x, y = split_xy(dataset,TRAIN_SIZE,'T (degC)',PREDICT_GAP)

print(x)
print(y)
print(x.shape, y.shape) # (420547, 360, 14) (420547,) -> (419687, 720, 14) (419687,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=2024, shuffle=True 
)

# scaler = StandardScaler
# scaler.fit(x_train)

# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# x_train = x_train.reshape(-1, 28, 2)
# x_test = x_test.reshape(-1, 28, 2)


print(x_train.shape, y_train.shape)  # (335749, 720, 14) (335749,)
print(x_test.shape, y_test.shape)  # (83938, 720, 14) (83938,)


#2. 모델 구성
model = Sequential()
# model.add(LSTM(100, input_shape = (720,14))) 
model.add(Conv1D(3, 50, input_shape = (720, 14)))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=10,
                   verbose=2,
                   restore_best_weights=True,
                   )
model.fit(x_train, y_train, epochs=1000, batch_size=128, verbose=2, callbacks=[es], validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("loss : ", results)
from sklearn.metrics import r2_score
print(y_test.shape, y_predict.shape)
r2_y_predict = r2_score(y_test, y_predict)
print('R2 : ', r2_y_predict)


# loss :  22.472246170043945
# (125907,) (125907, 1)
# R2 :  0.6807804861919331

