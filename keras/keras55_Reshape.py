import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Reshape, Conv1D, LSTM
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from keras.callbacks import EarlyStopping

# 손글씨 이미지셋
# 2차원 형태(28, 28), 784차원 공간의 한점
#1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) 3차원 (60000,) 1차원
print(x_test.shape, y_test.shape)# (10000, 28, 28) (10000,)

########## 스케일링 1-1 ############# 이미지에서는 어차피 255까지밖에 없음 // minmaxscaler // 부동소수 속도 빠르다
x_train = x_train/255.
x_test = x_test/255.


print(x_train[0])
print(y_train[0])
print(np.unique(y_train, return_counts=True))  
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#   dtype=int64))


x_train = x_train.reshape(60000, 28, 28, 1)  # 데이터 내용 순서 변화 없다
x_test = x_test.reshape(10000, 28, 28, 1)  # 데이터 내용 순서 변화 없다

print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)
print(y_train.shape, y_test.shape)  # (60000,) (10000,)

print(y_train[0])   # 0
print(y_train[1])   # 5
ohe = OneHotEncoder(sparse = False)
y_train = ohe.fit_transform(y_train.reshape(-1, 2))
y_test = ohe.fit_transform(y_test.reshape(-1, 2))
print(y_train[0])   # -1,1 일때 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
print(y_train[1])   # -1,1 일때 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
print(y_train[0])   # -1,2 일때 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
print(y_train[1])   # -1,2 일때 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]


print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)
# print(x_train[0])
# print(np.max(x_train), np.min(x_test))


# #2. 모델
# model = Sequential()
# model.add(Dense(9, input_shape=(28,28,1)))
# model.add(Conv2D(filters=10, kernel_size=(3,3)))    # 27,27,9
# model.add(Reshape(target_shape=(26*26, 10)))        # 연산 x 형태만 바꿈
# model.add(Conv1D(15, 4))
# model.add(LSTM(8, return_sequences=True))
# model.add(Conv1D(14, 2))  
# model.add(Flatten())
# model.add(Dense(8)) 
# model.add(Dense(7, activation='swish'))
# model.add(Dense(6, activation='swish'))
# model.add(Dense(10, activation='softmax'))          # N,10


# print(model.summary())
# # (kenel_size * channels + bias) * filters = 


""" #3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='acc',
                mode='min',
                patience=25,
                verbose=1,
                restore_best_weights=True
                )
model.fit(x_train, y_train, epochs=300, batch_size=1000, verbose=1, validation_split=0.2)

#4 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
y_predict = model.predict(x_test)
arg_pre = np.argmax(y_predict, axis=1)
arg_test = np.argmax(y_test, axis=1)
r2 = r2_score(arg_test, arg_pre)
print('r2 :', r2) """



'''
import matplotlib.pyplot as plt # 이미지 볼때,,
print(y_train[0:10])    # [5 0 4 1 9 2 1 3 1 4]
plt.imshow(x_train[0], 'PuBu')
plt.show()
'''


'''
# loss 0.18298788368701935
# acc 0.9818000197410583

'''







