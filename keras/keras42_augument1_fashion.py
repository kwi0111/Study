import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten ,Input, Dropout
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from keras.callbacks import EarlyStopping
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.


train_datagen = ImageDataGenerator(
    # rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.7,
    
    fill_mode='nearest',
    
)

augumet_size = 40000      # 가상 이미지의 수 (변수)

randidx = np.random.randint(x_train.shape[0], size=augumet_size)    # 60000중에서 40000개의 숫자를 뽑아내라.
                                                                    # np.random.randint(60000, 40000)

print(randidx)  # [15687 24705 24984 ... 28019 44512 18162]
print(np.min(randidx), np.max(randidx)) # 2 59998

x_augumented = x_train[randidx].copy()  # 메모리 원데이터에 영향을 미치지 않기위해서 사용
y_augumented = y_train[randidx].copy()   # 키벨류로 쌍이 맞음

# print(x_augumented)
# print(x_augumented.shape)   # (40000, 28, 28)
# print(y_augumented)
# print(y_augumented.shape)   # (40000,)

# x_augumented = x_augumented.reshape(-1, 28, 28, 1)

x_augumented = x_augumented.reshape(
    x_augumented.shape[0], x_augumented.shape[1], x_augumented.shape[2], 1)

x_augumented = train_datagen.flow(
    x_augumented, y_augumented,
    batch_size=augumet_size,
    shuffle=False,
    
).next()[0]    # 4만개 변환

print(x_augumented)
print(x_augumented.shape)   # (40000, 28, 28, 1)


print(x_train.shape)    #(60000, 28, 28)
x_train = x_train.reshape(60000, 28, 28, 1)   
x_test = x_test.reshape(10000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augumented))        # 사슬 같이 잇다
y_train = np.concatenate((y_train, y_augumented))        # 사슬 같이 잇다

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28, 1) (10000,)




print(x_train[0])
print(y_train[0])
print(np.unique(y_train, return_counts=True))  
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([10064,  9891,  9978,  9934,  9982,  9988, 10008, 10016, 10094,
print(pd.value_counts(y_test))
# 9    1000
# 2    1000
# 1    1000
# 6    1000
# 4    1000
# 5    1000
# 7    1000
# 3    1000
# 8    1000
# 0    1000
print(x_train.shape[0]) # 100000

ohe = OneHotEncoder(sparse = False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.fit_transform(y_test.reshape(-1, 1))



#2. 모델
model = Sequential()
model.add(Conv2D(9, (2,2), input_shape=(28, 28, 1),  strides=2, activation='relu')) 
                        # shape = (batch_size, rows, colums, channels) // 배치 디폴트 32
                        # shape = (batch_size, heights, widths, channels)
model.add(Conv2D(10, (3,3), strides=2, activation='relu'))   
model.add(Conv2D(15, (3,3), strides=2, activation='relu'))
model.add(Flatten())    
model.add(Dense(8, activation='relu')) 
model.add(Dense(7, activation='relu'))
model.add(Dense(5))
model.add(Dense(10, activation='softmax'))

print(model.summary())


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=5,
                   verbose=1,
                   restore_best_weights=True,
                   )
model.fit(x_train, y_train, epochs=200, batch_size=128, verbose=1, validation_split=0.2)

#4 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss', results[0])
print('acc', results[1])


'''

loss 0.448490709066391
acc 0.8384000062942505


loss 0.406208336353302
acc 0.8615000247955322


'''