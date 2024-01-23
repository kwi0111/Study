import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import cifar100
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

# train , test = cifar100.load_data()
# x_train, y_train = train

(x_train, y_train), (x_test, y_test) =  cifar100.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train, return_counts=True))

########## 스케일링 1-1 ############# 이미지에서는 어차피 255까지밖에 없음 // minmaxscaler // 부동소수 속도 빠르다
# x_train = x_train/255.
# x_test = x_test/255.

########## 스케일링 1-2 ############# 
# x_train = (x_train-127.5)/127.5
# x_test = (x_test-127.5)/127.5

########## 스케일링 2-1 #############
x_train = x_train.reshape(50000, 3072)
x_test = x_test.reshape(10000, 3072)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)

########## 스케일링 2-2 #############
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()
print(y_train.shape, y_test.shape)  # (50000, 100) (10000, 100)

# x_train = x_train.astype(np.float32)/255.0
# x_test = x_test.astype(np.float32)/255.0

#2. 모델 구성
input1 = Input(shape=(32, 32, 3))
Conv2D1 = Conv2D(10, (1,1), padding='same', activation="relu")(input1)
Conv2D2 = Conv2D(10, (1,1), padding='same')(Conv2D1)
drop1 = Dropout(0.2)(Conv2D2)
Flatten1 = Flatten()(drop1)
output1 = Dense(100, activation='softmax')(Flatten1)
model = Model(inputs = input1, outputs = output1) 


print(model.summary())

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=15,
                   verbose=1,
                   restore_best_weights=True,
                   )
model.fit(x_train, y_train, epochs=300, batch_size=64, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("loss : ", results[0])
print("acc : ", results[1])


'''
255
loss :  3.358400821685791
acc :  0.22990000247955322

127.5
loss :  3.329622983932495
acc :  0.22779999673366547

민맥스
loss :  3.4152114391326904
acc :  0.21850000321865082

스탠다드
loss :  3.3483169078826904
acc :  0.2190999984741211

'''