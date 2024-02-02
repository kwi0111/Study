from keras.datasets import imdb
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)  # 10000 단어 사전 갯수

print(x_train.shape, y_train.shape) # (25000,) (25000,)
print(x_test.shape, y_test.shape)   # (25000,) (25000,)
print(len(x_train[0]), len(x_test[0]))  # 218 68

# print(y_train[:20]) # [1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]
# print(np.unique(y_train, return_counts=True))   # array([0, 1], dtype=int64), array([12500, 12500]

# print("단어의 최대길이 : ", max(len(i) for i in x_train)) # 단어의 최대길이 :  2494
# print("단어의 평균길이 : ", sum(map(len, x_train)) / len(x_train)) # 단어의 평균길이 :  238.71364

# print(type(x_train))    # <class 'numpy.ndarray'>
# print(x_train[0])
# print(len(x_train[0]))  # 218

# 전처리 
from keras.utils import pad_sequences
use_text = 300

x_train = pad_sequences(x_train, padding='pre', maxlen=use_text, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=use_text, truncating='pre')
# print(x_train[0])
# print(x_train.shape, x_test.shape) # (25000, 230) (25000, 230)

#2. 모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, Flatten, LSTM, Dropout, SimpleRNN
model = Sequential()
model.add(Embedding(10000, 100, input_length=300))
# model.add(Conv1D(10, 2,  input_shape = (use_text, 1), activation='relu'))
# model.add(Flatten())
model.add(SimpleRNN(10, input_shape = (use_text, 1), activation='relu')) 
model.add(Dropout(0.05)) 
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=15,
                   verbose=1,
                   restore_best_weights=True)
model.fit(x_train, y_train, epochs=300, batch_size=64, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가 및 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("loss : ", results[0])
print("acc : ", results[1])
# print(y_predict[0])
# print(x_train[0])


'''
LSTM
loss :  0.5735107660293579
acc :  0.7087200284004211

SimpleRNN
loss :  0.3264234960079193
acc :  0.8676000237464905

Conv1D
loss :  0.2825596034526825
acc :  0.8823599815368652


'''



