# 오토인코더 비지도학습 (y가 없다)
# GAN에서 loss 의미가 없을수도 있다.


import numpy as np
from keras.datasets import mnist

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

#2. 모델
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input_img = Input(shape=(784,))
## 인코더
### 레이어 해보고 잘나오는거 쓰면된다.
encoded = Dense(512, activation='relu')(input_img)
# encoded = Dense(32, activation='sigmoid')(input_img)
# encoded = Dense(1, activation='relu')(input_img)
# encoded = Dense(1024, activation='relu')(input_img)

## 디코더
decoded = Dense(28*28, activation='sigmoid')(encoded)
# decoded = Dense(28*28, activation='linear')(encoded)
# decoded = Dense(28*28, activation='relu')(encoded)
# decoded = Dense(28*28, activation='tanh')(encoded)

autoencoder = Model(input_img, decoded)
autoencoder.summary()

#3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss='mse')
# autoencoder.compile(optimizer='adam', loss='mae')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=60, batch_size=32, validation_split=0.2)

#4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 7
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()




