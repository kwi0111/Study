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

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])  # 1 긍정, 0 부정


token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index) 
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화예요': 6, '추천하고': 7, 
# '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15,
# '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, 
# '재밋네요': 23, '상헌이': 24, '바보': 25, '반장': 26, '잘생겼다': 27, '욱이': 28, '또': 29, '잔다': 30} // 30 = 단어 사전의 개수

x = token.texts_to_sequences(docs)
print(x)   
# [[2, 3], [1, 4], [1, 5, 6],
# [7, 8, 9], [10, 11, 12, 13, 14], [15],
# [16], [17, 18], [19, 20],
# [21], [2, 22], [1, 23],
# [24, 25], [26, 27], [28, 29, 30]]
# 데이터 크기 일정하지 않음. // 크면 잘라내고 작으면 키운다 // 0넣는다 = cnn 패딩 // 패딩 통상적으로 왼쪽에 잡는다 // 
print(type(x))  # <class 'list'>
# x = np.array(x) #  차원이 달라서 에러뜸

from keras.utils import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5 , truncating='pre') # padding='post' 0이 뒤로간다. // maxlen=4 4개로 자른다. // truncating 앞에 자를거냐 뒤에 자를거냐
print(pad_x)
# print(pad_x.shape)  # (15, 5)
x = pad_x.reshape(-1, 5, 1)
y = labels
print(x) 
print(x.shape)  # (15, 5, 1)    # 2차원을 그냥 LSTM에 넣으면 알아서 돈다,,

# 임베딩에서 알아서 해준다.

word_size = len(token.word_index) + 1  # 31
print(word_size)


#2. 모델              
model = Sequential()   
###################################### 임베딩 1 
# model.add(Embedding(input_dim=31, output_dim=100, input_length=5))   # 입력 단어사전 수 31 백터화 , Dense 출력 노드 100, length = (15, 5)을 기준으로 삼겠다.
# 임베딩 연산량 = input_dim * output_dim = 31 * 100 =3100
# 임베딩 인풋의 shape : 2차원, 임베딩 아웃풋의 shape : 3차원 ?
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3100

#  lstm (LSTM)                 (None, 5, 100)            80400

#  lstm_1 (LSTM)               (None, 10)                4440

#  dense (Dense)               (None, 100)               1100

#  dense_1 (Dense)             (None, 10)                1010

#  dense_2 (Dense)             (None, 100)               1100

#  dense_3 (Dense)             (None, 1)                 101

############################################# 임베딩2
# model.add(Embedding(input_dim=31, output_dim=100))   # 입력 단어수 31 백터화 , Dense 출력 10, length = (15, 5)을 기준으로 삼겠다.
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, None, 100)         3100

#  lstm (LSTM)                 (None, None, 100)         80400

#  lstm_1 (LSTM)               (None, 10)                4440

#  dense (Dense)               (None, 100)               1100

#  dense_1 (Dense)             (None, 10)                1010

#  dense_2 (Dense)             (None, 100)               1100

#  dense_3 (Dense)             (None, 1)                 101

############################################# 임베딩3
# model.add(Embedding(input_dim=40, output_dim=100))   # 입력 단어수 31 백터화 , Dense 출력 10, length = (15, 5)을 기준으로 삼겠다.
# input_dim = 31 디폴트 <<<< 웬만하면 이거 쓰는게 낫다.
# input_dim = 20 단어사전의 갯수보다 작을때 : 연산량 줄어, 단어사전에서 임의로 뺀다 : 성능 조금 저하
# input_dim = 41 단어사전의 갯수보다 클때 : 연산량 늘어, 임의의 랜덤 데이터 임베딩 생성 : 성능 조금 저하

############################################# 임베딩4
model.add(Embedding(31, 100))   # 돌아간다.
# model.add(Embedding(31, 100, 5))  # 안돌아간다.
# model.add(Embedding(31, 100, input_length = 5))  # 돌아간다.
# model.add(Embedding(31, 100, input_length = 6))  # 안돌아간다.
# model.add(Embedding(31, 100, input_length = 1))  # 돌아간다.
# input_length = 2,3,4,6은 안돌아간다.



model.add(LSTM(100, return_sequences=True, input_shape = (5, 1), activation='relu'))  
model.add(LSTM(10, activation='relu'))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))

model.summary()


""" # #3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, y, epochs=500, batch_size=1, verbose=1)

# #4. 평가 및 예측
results = model.evaluate(x, y)
y_predict = model.predict(x)
print("loss : ", results[0])
print("acc : ", results[1]) """


# print(y_predict)
# loss :  2.5690333131933585e-05
# acc :  1.0














