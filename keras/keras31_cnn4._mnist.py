import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
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

########## 스케일링 1-2 ############# 
# x_train = (x_train-127.5)/127.5
# x_test = (x_test-127.5)/127.5

########## 스케일링 2-1 #############
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

########## 스케일링 2-2 #############
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# StandardScaler (x_train - 127.5) / 127.5  -> 이미지 [-1, 1]로 일반화

print(x_train[0])
print(y_train[0])
print(np.unique(y_train, return_counts=True))  
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#   dtype=int64))
print(pd.value_counts(y_test))
# 1    1135
# 2    1032
# 7    1028
# 3    1010
# 9    1009
# 4     982
# 0     980
# 8     974
# 6     958
# 5     892

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)# (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)  # 데이터 내용 순서 변화 없다
x_test = x_test.reshape(10000, 28, 28, 1)  # 데이터 내용 순서 변화 없다

print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) # 데이터 내용 순서 변화 없다
# print(x_train.shape[0]) # 60000

# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
# print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)
'''

ohe = OneHotEncoder(sparse = False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.fit_transform(y_test.reshape(-1, 1))

print(x_train[0])
print(np.max(x_train), np.min(x_test))



#2. 모델
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(28, 28, 1))) 
                        # shape = (batch_size, rows, colums, channels) // 배치 디폴트 32
                        # shape = (batch_size, heights, widths, channels)
model.add(Conv2D(filters=30, kernel_size=(2,2)))    # 4차원을 받아야함
model.add(Conv2D(20, (2,2), activation='swish'))
model.add(Conv2D(filters=30, kernel_size=(2,2)))  
model.add(Flatten())    # 입력 데이터를 1차원으로 평탄화. 2D 혹은 3D의 특징 맵(feature map)을 1D 벡터로 변환, 이후의 레이어에서 처리하기 쉽게 만들어주는 역할
model.add(Dense(units=20)) # 주로 2차원 받음 
model.add(Dense(80, activation='swish', input_shape=(20, )))
#                   shape=(batch_size, input_dim)
model.add(Dense(20, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(10, activation='softmax'))


print(model.summary())
# (kenel_size * channels + bias) * filters = 

 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='acc',
                mode='min',
                patience=25,
                verbose=1,
                restore_best_weights=True
                )
model.fit(x_train, y_train, epochs=300, batch_size=2000, verbose=1, validation_split=0.2)

#4 평가, 예측   acc = 0.98 이상
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
y_predict = model.predict(x_test)
arg_pre = np.argmax(y_predict, axis=1)
arg_test = np.argmax(y_test, axis=1)
r2 = r2_score(arg_test, arg_pre)
print('r2 :', r2)



import matplotlib.pyplot as plt # 이미지 볼때,,
print(y_train[0:10])    # [5 0 4 1 9 2 1 3 1 4]
plt.imshow(x_train[0], 'PuBu')
plt.show()
'''


'''
# loss 0.18298788368701935
# acc 0.9818000197410583



'''

