import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import cifar100
from sklearn.preprocessing import OneHotEncoder

# train , test = cifar100.load_data()
# x_train, y_train = train

(x_train, y_train), (x_test, y_test) =  cifar100.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train, return_counts=True))

# print("y_train[2]",y_train[2])

# import matplotlib.pyplot as plt
# plt.imshow(x_train[2])
# plt.show()
ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()
print(y_train.shape, y_test.shape)  # (50000, 100) (10000, 100)

x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0

#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2, 2), padding='same', activation='swish', 
                 input_shape=(32, 32, 3)))
model.add(Conv2D(50, (2, 2),padding='same', activation='swish'))
model.add(MaxPooling2D(pool_size=(2, 2)))   
model.add(Dropout(0.3))
model.add(Conv2D(100, (2, 2),padding='same', activation='swish'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(50, (2, 2),padding='same', activation='swish'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(100, (2, 2), padding='same', activation='swish'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(50, (2, 2), padding='same', activation='swish'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(10, activation='swish'))
model.add(Dense(100, activation='softmax'))

# 맥스 풀링 : 
# 패딩 : 
# 스트라이드 : 



print(model.summary())

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='acc',
                   mode='auto',
                   patience=20,
                   verbose=1,
                   restore_best_weights=True,
                   )
model.fit(x_train, y_train, epochs=600, batch_size=500, verbose=1, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("loss : ", results[0])
print("acc : ", results[1])


