import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


#1. 데이터
path = "c://_data//kaggle//jena//"

dataset = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

size = 4
def split_xy(dataset, size, y_col):     
    result_x = []
    result_y = []
    for i in range(len(dataset) - size):
        result_x.append(dataset[i:i+size])
        y_row = dataset.iloc[i+size]
        result_y.append(y_row[y_col])
    return np.array(result_x), np.array(result_y)

x, y = split_xy(dataset, size, 'T (degC)')
print(x)
print(y)
print(x.shape, y.shape) # (420547, 4, 14) (420547,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=2024, shuffle=True 
)

# scaler = StandardScaler
# scaler.fit(x_train)

# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# x_train = x_train.reshape(-1, 28, 2)
# x_test = x_test.reshape(-1, 28, 2)


print(x_train.shape, y_train.shape)  # (378492, 4, 14) (378492,)
print(x_test.shape, y_test.shape)  # (42055, 4, 14) (42055,)


#2. 모델 구성
model = Sequential()
model.add(LSTM(100, input_shape = (4,14))) 
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=15,
                   verbose=1,
                   restore_best_weights=True,
                   )
model.fit(x_train, y_train, epochs=1000, batch_size=3024, verbose=1, callbacks=[es], validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("loss : ", results)
from sklearn.metrics import r2_score
print(y_test.shape, y_predict.shape)
r2_y_predict = r2_score(y_test, y_predict)
print('R2 : ', r2_y_predict)

# loss :  0.127421572804451
# (84110,) (84110, 1)
# R2 :  0.9982194176382211


# def split_x(dataset, size):     
#     aaa = []
#     for i in range(len(dataset) - size + 1):
#         # subset = dataset[i : (i + size)]
#         # aaa.append(subset)
#         aaa.append(dataset[i:i+size])
#     return np.array(aaa)

# bbb = split_x(dataset, size)      # a로 bbb를 만듦.
# x = bbb[:-1]
# print(bbb.shape,x.shape)


# print(y.shape)  #(420551,)




# print(train_csv.info())
#  #   Column           Non-Null Count   Dtype
# ---  ------           --------------   -----
#  0   p (mbar)         420551 non-null  float64
#  1   T (degC)         420551 non-null  float64
#  2   Tpot (K)         420551 non-null  float64
#  3   Tdew (degC)      420551 non-null  float64
#  4   rh (%)           420551 non-null  float64
#  5   VPmax (mbar)     420551 non-null  float64
#  6   VPact (mbar)     420551 non-null  float64
#  7   VPdef (mbar)     420551 non-null  float64
#  8   sh (g/kg)        420551 non-null  float64
#  9   H2OC (mmol/mol)  420551 non-null  float64
#  10  rho (g/m**3)     420551 non-null  float64
#  11  wv (m/s)         420551 non-null  float64
#  12  max. wv (m/s)    420551 non-null  float64
#  13  wd (deg)         420551 non-null  float64





