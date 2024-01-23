import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.preprocessing import OneHotEncoder


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) = (60000, 28, 28, 1) 1생략,,
print(x_test.shape, y_test.shape    )# (10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])
print(np.unique(y_train, return_counts=True))  
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#   dtype=int64))
print(pd.value_counts(y_test))
x_train = x_train.reshape(60000, 28, 28, 1) # 데이터 내용 순서 변화 없다
# x_test = x_test.reshape(10000, 28, 28, 1) # 데이터 내용 순서 변화 없다
# print(x_train.shape[0]) # 60000

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) # 데이터 내용 순서 변화 없다
print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

ohe = OneHotEncoder(sparse = False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.fit_transform(y_test.reshape(-1, 1))


#2. 모델
model = Sequential()
model.add(Conv2D(9, (2,2), 
                 strides=1,
                 input_shape=(28, 28, 1),
                 padding='same',))
                        # stride : 커널의 보폭
                        # 모양 같이 하고 싶으면 padding : 'same'쓴다. 'valid'은 디폴트
                        # shape = (batch_size, rows, colums, channels) // 배치 디폴트 32
                        # shape = (batch_size, heights, widths, channels)
model.add(Conv2D(filters=10, kernel_size=(3,3),padding='same'))    # 4차원을 받아야함
model.add(Conv2D(15, (4,4), padding='same') )
model.add(Flatten())    # 입력 데이터를 1차원으로 평탄화. 2D 혹은 3D의 특징 맵(feature map)을 1D 벡터로 변환, 이후의 레이어에서 처리하기 쉽게 만들어주는 역할 // reshape랑 동일 개념
model.add(Dense(units=8)) # 주로 2차원 받음 
model.add(Dense(7, input_shape=(8, )))
#                   shape=(batch_size, input_dim)
model.add(Dense(6))
model.add(Dense(10, activation='softmax'))


print(model.summary())

# (kenel_size * channels + bias) * filters = 

 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2)

#4 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss', results[0])
print('acc', results[1])
y_predict = model.predict(x_test)
arg_pre = np.argmax(y_predict, axis=1)
arg_test = np.argmax(y_test, axis=1)

print(arg_pre)


import matplotlib.pyplot as plt # 이미지 볼때,,
plt.imshow(x_train[1], 'PuBu')
plt.show()


# loss 0.3599582314491272
# acc 0.899399995803833
