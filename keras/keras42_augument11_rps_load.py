import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Dropout, Conv2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

 
#1. 데이터
np_path = 'c:/_data/_save_npy/'

x_train = np.load(np_path + 'keras42_10_x_train.npy')
x_test = np.load(np_path + 'keras42_10_x_test.npy')
y_train = np.load(np_path + 'keras42_10_y_train.npy')
y_test = np.load(np_path + 'keras42_10_y_test.npy')

#2. 모델
model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', strides=2, input_shape = (150,150,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), padding='same',  strides=2, activation='relu' ))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(62, (2,2), padding='same',  strides=2, activation='relu' ))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(16, activation='relu')) 
model.add(Dense(3, activation='softmax')) 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='auto', patience=30, verbose=1, restore_best_weights=True)
model.fit(x_train,
          y_train,
          epochs=100,
          batch_size=10,
          verbose=1,
          validation_split=0.2,
          )

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("loss : ", results[0])
print("acc : ", results[1])

# loss :  0.0019408464431762695
# acc :  1.0




