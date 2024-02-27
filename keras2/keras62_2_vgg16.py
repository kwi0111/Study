import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import cifar10
import tensorflow as tf
from keras.applications import VGG16

tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)   # 2.9.0

# # CIFAR-10 데이터셋 로드
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

# VGG16 모델 불러오기
vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32,32,3))
vgg16.trainable = False # 가중치 동결

# 모델 빌드
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()























