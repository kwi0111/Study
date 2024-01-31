from keras.datasets import cifar10
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, LSTM, Conv1D
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 3차원 형태(32, 32, 3), 3072차원 공간의 한점
#1.데이터
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
x_train = x_train.reshape(-1,96,32)
x_test = x_test.reshape(-1,96,32)

ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = ohe.fit_transform(y_test.reshape(-1,1)).toarray()
print(y_train.shape)
print(x_train.shape)
# (50000, 10)
# (50000, 32, 32, 3)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(7, 2, padding='same', input_shape = (96,32)))
model.add(Conv1D(10, 2, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax')) 

print(model.summary())


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=25,
                verbose=1,
                restore_best_weights=True
                )
model.fit(x_train, y_train, epochs=300, batch_size=2000, validation_split=0.2, verbose=1, callbacks=[es])

#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("loss : ", results[0] )
print("acc : ", results[1] )



'''

import matplotlib.pyplot as plt
plt.imshow(x_train[2])
plt.show()
loss :  0.6670249104499817
acc :  0.7738000154495239

LSTM
loss :  1.391109824180603
acc :  0.5131999850273132

Conv1D
loss :  2.302605152130127
acc :  0.10000000149011612

'''