# 01~11까지를 cnn으로 만들어서 성능 비교

# 보스턴에 관한 데이터

import numpy as np
import pandas as pd
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston


#1. 데이터
datasets = load_boston()    
print(datasets)
x = datasets.data         
y = datasets.target       
print(x.shape, y.shape)  #(506, 13), (506,) -> 

print(datasets.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,   
                                                    train_size=0.7,
                                                    random_state=123,     
                                                    shuffle=True,
                                                    )

# from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
# from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# scaler = MinMaxScaler() # 클래스 정의
# scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의

# y = y.reshape(506, 1)


# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)

x_train = x_train.reshape(-1, 13, 1, 1)
x_test = x_test.reshape(-1, 13, 1, 1)



# print(x.shape, y.shape)  # (506, 13, 1, 1) (506, 1)


# ohe = OneHotEncoder()
# y = ohe.fit_transform(y.reshape(-1,1)).toarray()


model = Sequential()
model.add(Conv2D(10, kernel_size=(1, 1),  activation='relu', 
                 input_shape=(13, 1, 1)))
# model.add(Conv2D(50, (1, 1), activation='relu'))
# model.add(Dropout(0.3))
# model.add(Conv2D(100, (1, 1), activation='relu'))
# model.add(Conv2D(50, (1, 1), activation='relu'))
# model.add(Dropout(0.3))
# model.add(Conv2D(100, (1, 1), activation='relu'))
# model.add(Conv2D(50, (1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

print(model.summary())

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='loss',
                   mode='auto',
                   patience=15,
                   verbose=1,
                   restore_best_weights=True,
                   )
model.fit(x_train, y_train, epochs=300, batch_size=30, verbose=1, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("loss : ", results[0])
print("acc : ", results[1])

'''

# loss :  0.5976815819740295
# acc :  0.8970999717712402

CNN
loss :  0.004355018027126789
acc :  0.9956331849098206


'''