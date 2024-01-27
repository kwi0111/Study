from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten ,Input, Dropout, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
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

augumet_size = 15000      # 가상 이미지의 수 (변수)

randidx = np.random.randint(x_train.shape[0], size=augumet_size)    # 50000중에서 15000개의 숫자를 뽑아내라.
                                                                    # np.random.randint(60000, 20000)
print(randidx)  # [  872 23666  7905 ... 17524 37111 22648]
print(np.min(randidx), np.max(randidx)) # 4 49999
x_augumented = x_train[randidx].copy()  # 메모리 원데이터에 영향을 미치지 않기위해서 사용 // 안전한 작업 수행
y_augumented = y_train[randidx].copy()   # 키벨류로 쌍이 맞음
print(x_augumented)
print(x_augumented.shape)   # (15000, 32, 32, 3)
print(y_augumented)         # [7 7 5 ... 2 7 1]
print(y_augumented.shape)   # (15000, 1)

# x_augumented = x_augumented.reshape(-1, 28, 28, 1)
# x_augumented = x_augumented.reshape(
#     x_augumented.shape[0], x_augumented.shape[1], x_augumented.shape[2], 3)

x_augumented = train_datagen.flow(
    x_augumented, y_augumented,
    batch_size=augumet_size,
    shuffle=False,
    
).next()[0]    #

print(x_augumented)
print(x_augumented.shape)   # (15000, 32, 32, 3)


print(x_train.shape)    # (50000, 32, 32, 3)
# x_train = x_train.reshape(-1, 32, 32, 3)   
# x_test = x_test.reshape(-1, 32, 32, 1)

x_train = np.concatenate((x_train, x_augumented))        # 사슬 같이 잇다
y_train = np.concatenate((y_train, y_augumented))        # 사슬 같이 잇다

print(x_train.shape, y_train.shape) # (65000, 32, 32, 3) (65000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 1) (10000, 1)


ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = ohe.transform(y_test.reshape(-1,1)).toarray()
print(y_train.shape, y_test.shape)  # (65000, 100) (10000, 100)
unique, counts = (np.unique(y_test, return_counts=True))
print(unique)   # [0. 1.]
print(counts)   # [990000  10000]


#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(32, 32, 3), padding='same', strides=2, activation='relu')) 
                        # shape = (batch_size, rows, colums, channels) // 배치 디폴트 32
                        # shape = (batch_size, heights, widths, channels)
model.add(MaxPooling2D())
model.add(Dropout(0.1))
model.add(Conv2D(16, (2,2), padding='same', strides=2, activation='relu')) 
model.add(MaxPooling2D())
model.add(Dropout(0.1))
# model.add(Conv2D(4, (2,2), padding='same', strides=2, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.1))
model.add(Flatten())    
model.add(Dense(7, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(100, activation='softmax'))



print(model.summary())

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=10,
                   verbose=1,
                   restore_best_weights=True,
                   )
model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("loss : ", results[0])
print("acc : ", results[1])

'''
loss :  1.4143774509429932
acc :  0.47909998893737793
'''
'''
loss :  3.5994749069213867
acc :  0.1324000060558319

'''