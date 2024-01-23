from keras.datasets import cifar10
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
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

########## 스케일링 1-1 ############# 이미지에서는 어차피 255까지밖에 없음 // minmaxscaler // 부동소수 속도 빠르다
# x_train = x_train/255.
# x_test = x_test/255.

########## 스케일링 1-2 ############# 
# x_train = (x_train-127.5)/127.5
# x_test = (x_test-127.5)/127.5

########## 스케일링 2-1 #############
x_train = x_train.reshape(-1, 3072)
x_test = x_test.reshape(-1, 3072)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 32, 32 ,3)
x_test = x_test.reshape(-1, 32, 32, 3)

########## 스케일링 2-2 #############
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = ohe.fit_transform(y_test.reshape(-1,1)).toarray()
print(y_train.shape)
print(x_train.shape)
# (50000, 10)
# (50000, 32, 32, 3)


#2. 모델
input1 = Input(shape=(32, 32, 3))
Conv2D1 = Conv2D(10, (1,1), padding='same', activation="relu")(input1)
Conv2D2 = Conv2D(10, (1,1), padding='same')(Conv2D1)
drop1 = Dropout(0.2)(Conv2D2)
Flatten1 = Flatten()(drop1)
output1 = Dense(10, activation='softmax')(Flatten1)
model = Model(inputs = input1, outputs = output1) 

print(model.summary())


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=15,
                verbose=1,
                restore_best_weights=True
                )
model.fit(x_train, y_train, epochs=300, batch_size=64, validation_split=0.2, verbose=1, callbacks=[es])

#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("loss : ", results[0] )
print("acc : ", results[1] )



'''

# import matplotlib.pyplot as plt
# plt.imshow(x_train[2])
# plt.show()
# loss :  0.6670249104499817
# acc :  0.7738000154495239

255
loss :  1.5469030141830444
acc :  0.45170000195503235


127.5
loss :  1.5166871547698975
acc :  0.46779999136924744

민맥스 
loss :  1.5499435663223267
acc :  0.45750001072883606

스탠다드
loss :  1.5481995344161987
acc :  0.45719999074935913

'''

