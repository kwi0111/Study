import numpy as np
import pandas as pd
import time
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Input, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

#1.데이터
train_datagen = ImageDataGenerator(rescale=1./255,
                                   )

path_train = 'c:/_data/image/rps/'

xy_train = train_datagen.flow_from_directory(path_train,
                                             target_size=(150,150),
                                             batch_size=10000,
                                             class_mode='categorical',
                                             shuffle=True,
                                             )
print(xy_train) # <keras.preprocessing.image.DirectoryIterator object at 0x0000027E9F3EFC40>
# print(xy_train[0][0])
# print(xy_train[0][1])

x = xy_train[0][0]
y = xy_train[0][1]
# print(np.unique(y, return_counts=True))
# (array([0., 1.], dtype=float32), array([5040, 2520], dtype=int64))

# print(pd.value_counts(y)) #안먹음

# print(x.shape, y.shape) # (1027, 300, 300, 3) (1027,)

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    train_size=0.85,
                                                    random_state=2024,
                                                    shuffle=True,
                                                    stratify=y,
                                                    )

np_path = 'c:/_data/_save_npy/'
np.save(np_path + 'keras39_9_x_train.npy', arr = x_train)
np.save(np_path + 'keras39_9_y_train.npy', arr = y_train)
np.save(np_path + 'keras39_9_x_test.npy', arr = x_test)
np.save(np_path + 'keras39_9_y_test.npy', arr = y_test)



