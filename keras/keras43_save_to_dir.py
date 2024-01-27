
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
    save_to_dir='c:\\_data\\temp\\' # 폴더에 증폭된 데이터 생성
).next()[0]    # 4만개 변환









