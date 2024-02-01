from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Embedding

#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', ' 참 잘만든 영화예요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다.', ' 참 재밋네요',
    '상헌이 바보', '반장 잘생겼다', '욱이 또 잔다',
    
]
x_predict = ['나는 정룡이가 정말 싫다. 재미없다 너무 정말']
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])  # 1 긍정, 0 부정


token = Tokenizer()
token.fit_on_texts(docs)
token.fit_on_texts(x_predict)
# print(token.word_index) 
# {'너무': 1, '정말': 2, '참': 3, '재미없다': 4, '나는': 5, '정룡이가': 6, '싫다': 7, '재미있다': 8, 
# '최고에요': 9, '잘만든': 10, '영화예요': 11, '추천하고': 12, '싶은': 13, '영화입니다': 14, '한': 15, '번': 16, 
# '더': 17, '보고': 18, '싶어요': 19, '글쎄': 20, '별로에요': 21, '생각보다': 22, '지루해요': 23, '연기가': 24, '어색해요': 25, '재미없어요': 26, 
# '재밋네요': 27, '상헌이': 28, '바보': 29, '반 장': 30, '잘생겼다': 31, '욱이': 32, '또': 33, '잔다': 34}
x = token.texts_to_sequences(docs)
x_predict = token.texts_to_sequences(x_predict)
# x1 = x + x_predict
# token.fit_on_texts(x1)
# print(type(x1))  # <class 'list'>

print(x)
# [[1, 5], [2, 6], [2, 7, 8], [9, 10, 11], [12, 13, 14, 15, 16], [17], [18], [19, 20],
# [21, 22], [23], [1, 3], [2, 24], [25, 26], [27, 28], [29, 30, 31]]
# print(x1)
# # [[1, 5], [2, 6], [2, 7, 8], [9, 10, 11], [12, 13, 14, 15, 16], [17], [18], [19, 20], 
# # [21, 22], [23], [1, 3], [2, 24], [25, 26], [27, 28], [29, 30, 31], [32, 33, 4, 34, 3, 1, 4]]
print(x_predict)    # [[32, 33, 4, 34, 3, 1, 4]]

print(type(x_predict))  # <class 'list'>
# x = np.array(x) #  차원이 달라서 에러뜸


from keras.utils import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=7 , truncating='pre')
# pad_x1 = pad_sequences(x1, padding='pre', maxlen=7 , truncating='pre')
pad_x_pred = pad_sequences(x_predict, padding='pre', maxlen=7 , truncating='pre')
# print(pad_x1)
print(pad_x_pred)
print(pad_x.shape)  # (15, 7)
x = pad_x.reshape(-1, 7, 1)
y = labels
# print(x) 

word_size = len(token.word_index) + 1  # 35
print(word_size)
print(x.shape)  # (15, 7, 1)    # 2차원을 그냥 LSTM에 넣으면 알아서 돈다,,

# 임베딩에서 알아서 해준다.



#2. 모델              
model = Sequential()   
model.add(Embedding(word_size, 100))
model.add(LSTM(100, return_sequences=True, input_shape = (7, 1), activation='relu'))  
model.add(LSTM(10, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# #3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x, y, epochs=100, batch_size=1, verbose=1)
#4. 평가 및 예측
results = model.evaluate(x, y)
y_predict = model.predict(x_predict)
print("loss : ", results[0])
print("acc : ", results[1]) 
print(y_predict)





############### 실습 ################
# x_predict = '나는 정룡이가 정말 싫다. 재미없다 너무 정말'


#결과 긍정? 부정?
# loss :  1.8509492747398326e-06
# acc :  1.0
# [[0.77458566]]










