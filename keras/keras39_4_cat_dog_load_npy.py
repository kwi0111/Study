import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten ,Input, Dropout, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from keras.callbacks import EarlyStopping
import time
from sklearn.model_selection import train_test_split
import os


#1. 데이터 // 땡겨오는 시간만 좀 걸리지 연산속도는 빠름
np_path = 'c:/_data/_save_npy/'
# np.save(np_path + 'keras39_1_x_train.npy', arr=xy_train[0][0])    # 넘파이 형태로 쏙 들어간다.
# np.save(np_path + 'keras39_1_y_train.npy', arr=xy_train[0][1])    
# np.save(np_path + 'keras39_1_x_test.npy', arr=xy_test[0][0])    
# np.save(np_path + 'keras39_1_y_test.npy', arr=xy_test[0][1])    

x_train = np.load(np_path + 'keras39_3_x_train.npy')
y_train = np.load(np_path + 'keras39_3_y_train.npy')
x_test = np.load(np_path + 'keras39_3_x_test.npy')
y_test = np.load(np_path + 'keras39_3_y_test.npy')

# print(x_train)
# print(x_train.shape,y_train.shape)    # 
# print(x_test.shape, y_test.shape)    # 

#2. 모델
model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', strides=2, input_shape = (140,140,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), padding='same',  strides=2, activation='relu' ))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), padding='same',  strides=2, activation='relu' ))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(16, activation='relu')) 
model.add(Dense(1, activation='sigmoid')) 

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=30,
                verbose=1,
                restore_best_weights=True
                )
model.fit(x_train,
          y_train,
          epochs=5,
                    # steps_per_epoch=16, # 전체 데이터 / batch = 160 / 10 = 16 /// 17이면 에러, 15면 나머지 소실
                    batch_size=32,    # fit_generator에서는 에러, fit에서는 안먹힘.
                    verbose=1,
                    validation_split=0.2, # 에러 
                    # validation_data=xy_test,
                    )

#4 평가, 예측
results = model.evaluate(x_test, y_test) 
submission = model.predict(x_test)
submission = submission.flatten()
print('loss', results[0])
print('acc', results[1])

failname = os.listdir('c:/_data/cat_and_dog//Test1/Test/')
failname[0] = failname[0].replace(".jpg","")

len(failname)
print(len(failname))   # x가 5천개다.

for i in range(len(failname)):
    failname[i] = failname[i].replace(".jpg","")

print(len(failname), len(submission))


submission_df = pd.DataFrame({"Id":failname, "Target":submission})
submission_df.to_csv("C:\_data\kaggle\catdog" + "submission_0124.csv", index=False)


