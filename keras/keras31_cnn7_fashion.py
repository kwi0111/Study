from keras. datasets import fashion_mnist
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test) =  fashion_mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)
print(np.unique(y_train, return_counts=True))

# print("y_train[2]",y_train[2])

# import matplotlib.pyplot as plt
# plt.imshow(x_train[2])
# plt.show()
ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = ohe.transform(y_test.reshape(-1,1)).toarray()
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

# x_train = x_train.astype(np.float32)/255.0
# x_test = x_test.astype(np.float32)/255.0

#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2, 2), padding='same', strides=2,  activation='swish', 
                 input_shape=(28, 28, 1)))
model.add(Conv2D(50, (2, 2),padding='same', strides=2, activation='swish'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(100, (2, 2),padding='same', strides=2, activation='swish'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(50, (2, 2),padding='same', activation='swish'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(100, (2, 2), padding='same', activation='swish'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(50, (2, 2), padding='same', activation='swish'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(10, activation='swish'))
model.add(Dense(10, activation='softmax'))

print(model.summary())

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='loss',
                   mode='auto',
                   patience=15,
                   verbose=1,
                   restore_best_weights=True,
                   )
model.fit(x_train, y_train, epochs=300, batch_size=64, verbose=1, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("loss : ", results[0])
print("acc : ", results[1])

'''
loss :  0.250508189201355
acc :  0.921999990940094
'''