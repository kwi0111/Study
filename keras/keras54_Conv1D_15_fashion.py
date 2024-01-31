from keras. datasets import fashion_mnist
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input, LSTM, Conv1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

(x_train, y_train), (x_test, y_test) =  fashion_mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)
print(np.unique(y_train, return_counts=True))

########## 스케일링 1-1 ############# 이미지에서는 어차피 255까지밖에 없음 // minmaxscaler // 부동소수 속도 빠르다
# x_train = x_train/255.
# x_test = x_test/255.

########## 스케일링 1-2 ############# 
# x_train = (x_train-127.5)/127.5
# x_test = (x_test-127.5)/127.5

########## 스케일링 2-1 #############
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 28*7, 4)
x_test = x_test.reshape(-1, 28*7, 4)

########## 스케일링 2-2 #############
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = ohe.transform(y_test.reshape(-1,1)).toarray()
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

# x_train = x_train.astype(np.float32)/255.0
# x_test = x_test.astype(np.float32)/255.0

#2. 모델 구성
model = Sequential()
model.add(Conv1D(7, 2, padding='same', input_shape = (28*7, 4)))
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
                   mode='auto',
                   patience=15,
                   verbose=1,
                   restore_best_weights=True,
                   )
model.fit(x_train, y_train, epochs=300, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("loss : ", results[0])
print("acc : ", results[1])

'''
배치 128
loss :  0.250508189201355
acc :  0.921999990940094

배치 64
loss :  0.2451872080564499
acc :  0.9164000153541565

배치32
loss :  0.27043017745018005
acc :  0.9074000120162964

배치 128 
loss :  0.25525301694869995
acc :  0.9190999865531921

loss :  0.2671722173690796
acc :  0.9300000071525574

255
loss :  0.3899014890193939
acc :  0.8637999892234802

127.5
loss :  0.3835061490535736
acc :  0.8626999855041504

민맥스
loss :  0.38703712821006775
acc :  0.8636999726295471


스탠다드
loss :  0.4063112437725067
acc :  0.8611999750137329

LSTM
loss :  0.39822497963905334
acc :  0.8585000038146973

Conv1D
loss :  0.33986344933509827
acc :  0.8835999965667725
'''