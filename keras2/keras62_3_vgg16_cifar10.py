# cifar10으로 모델 완성
# 성능비교, 시간 체크

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.datasets import cifar10
import tensorflow as tf
from keras.applications import VGG16
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
import time

tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)   # 2.9.0

# CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

x_train = x_train / 255.0
x_test = x_test / 255.0

# ohe = OneHotEncoder()
# y_train = ohe.fit_transform(y_train.reshape(-1,1)).toarray()
# y_test = ohe.transform(y_test.reshape(-1,1)).toarray()

# VGG16 모델 불러오기
vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32,32,3))
vgg16.trainable = False # 가중치 동결

# 모델 빌드
model = Sequential()
model.add(Input(shape=(32,32,3)))
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=25,
                verbose=1,
                restore_best_weights=True
                )
start_time = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=516, validation_split=0.2, verbose=1, callbacks=[es])
end_time = time.time()

#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("loss : ", results[0] )
print("acc : ", results[1] )
# model.summary()

# loss :  1.2024171352386475
# acc :  0.5875999927520752









