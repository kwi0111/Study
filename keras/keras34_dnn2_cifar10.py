from keras.datasets import cifar10
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
unique, counts = (np.unique(y_test, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
#       dtype=int64))
print(unique)
print(counts)
x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)
print(x_test.shape, x_train.shape)   # (10000, 3072) (50000, 3072)

x_train = x_train / 255.0
x_test = x_test / 255.0

ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = ohe.fit_transform(y_test.reshape(-1,1)).toarray()
print(y_train.shape)
print(x_train.shape)



#2. 모델
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=3072))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))


# model.add(Conv2D(filters=100, kernel_size=(2, 2), padding='same', strides=2, activation='swish', 
#                  input_shape=(32, 32, 3)))
# model.add(Conv2D(50, (2, 2),padding='same', strides=2, activation='swish'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))
# model.add(Conv2D(100, (2, 2),padding='same', strides=2, activation='swish'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(50, (2, 2),padding='same', strides=2, activation='swish'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))
# model.add(Conv2D(100, (2, 2), padding='same', strides=2, activation='swish'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(50, (2, 2), padding='same', strides=2, activation='swish'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dropout(0.3))
# model.add(Dense(10, activation='swish'))
# model.add(Dense(10, activation='softmax'))

# print(model.summary())


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=25,
                verbose=1,
                restore_best_weights=True
                )
model.fit(x_train, y_train, epochs=300, batch_size=516, validation_split=0.2, verbose=1, callbacks=[es])

#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("loss : ", results[0] )
print("acc : ", results[1] )

'''
loss :  1.3514968156814575
acc :  0.5224999785423279

'''

