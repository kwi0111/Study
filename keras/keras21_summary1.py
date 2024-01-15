from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_shape = (1,)))
model.add(Dense(4,))
model.add(Dense(2,))
model.add(Dense(1,))    # , 있어도댐

model.summary()





