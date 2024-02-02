from keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=1000,
                                                         test_split=0.2
                                                         )

print(x_train[0])
print(y_train[0])

print(x_train.shape, x_test.shape)    # (8982,) (2246,)
print(y_train.shape, y_test.shape)    # (8982,) (2246,)
# print(type(x_train))    # <class 'numpy.ndarray'>

print(y_train)  # [ 3  4  3 ... 25  3 25] 46개의 카테고리
# print(len(np.unique(y_train)))  # 46 : 0부터 45

# print(type(x_train))        # <class 'numpy.ndarray'>
# print(type(x_train[0]))     # <class 'list'>
# print(x_train[0])
# print(len(x_train[0]), len(x_train[1]))     # 87 56

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)) # 2376
print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train)) # 145.53

print(y_train)

# 전처리
from keras.utils import pad_sequences
max_text = 100
x_train = pad_sequences(x_train, padding='pre', maxlen=max_text, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=max_text, truncating='pre')

# y 원핫 or sparse_categorical_crossentropy
print(x_train[1])
print(x_train.shape, x_test.shape)  # (8982, 100) (2246, 100)
# x_train = x_train.reshape(-1, max_text, 1)
# x_test = x_test.reshape(-1, max_text, 1)
# print(x_train.shape, x_test.shape)  # (8982, 100, 1) (2246, 100, 1)
# from keras. preprocessing.text import Tokenizer

#2. 모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, Flatten, LSTM, Dropout
model = Sequential()
model.add(Embedding(1000, 100, input_length=100))
model.add(Conv1D(10, 2, input_shape = (max_text, 1)))
model.add(Flatten())
# model.add(LSTM(10))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.03))
model.add(Dense(100, activation='relu'))
model.add(Dense(46, activation='softmax'))

# model = Sequential()   
# model.add(Embedding(1000, 100))
# model.add(LSTM(100, return_sequences=True, input_shape = (max_text, 1), activation='relu'))  
# model.add(LSTM(10, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(46, activation='softmax'))

model.summary()

# #3. 컴파일, 훈련
from keras.callbacks import EarlyStopping
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=30,
                   verbose=1,
                   restore_best_weights=True)
model.fit(x_train, y_train, epochs=300, batch_size=20, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가 및 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("loss : ", results[0])
print("acc : ", results[1]) 
# print(y_predict[0])
# print(x_train[0])
# print(y_train[0])

'''

loss :  1.4351187944412231
acc :  0.6406945586204529

'''
