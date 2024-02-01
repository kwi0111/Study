from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd

text1 = '나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
text2 = '상헌이가 선생을 괴롭힌다. 상헌이는 못생겼다. 상헌이는 마구 마구 못생겼다.' 
# 띄어쓰기 단위로 수치화 // 어절 단위로 했다. // 자연어 데이터 수치화 // 
# 수치화 
token = Tokenizer()
token.fit_on_texts([text1, text2])

print(token.word_index) # 많이 먼저 나온 순서를 수치화
# {'마구': 1, '진짜': 2, '매우': 3, '상헌이는': 4, '못생겼다': 5, '나는': 6, '맛있는': 7, '밥을': 8, '엄청': 9, '먹었다': 10, '상헌이가': 11, '선생을': 12, '괴롭힌다': 13}

print(token.word_counts)    # 개수
# OrderedDict([('나는', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 5), ('먹었다', 1), ('상헌이가', 1), ('선생을', 1), ('괴롭힌다', 1), ('상헌이는', 2), ('못생 겼다', 2)])

x, y = token.texts_to_sequences([text1, text2]) # 텍스트를 정수 시퀀스로 변환
print(x)    # [[6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 10]
print(y)    # [11, 12, 13, 4, 5, 4, 1, 1, 5]

x = np.array(x).reshape(-1, 1)  # 리스트라서 넘파이에 reshape기능으로 해야함
y = np.array(y).reshape(-1, 1)  # 리스트라서 넘파이에 reshape기능으로 해야함
print(x.shape)  # (12, 1)
print(y.shape)  # (9, 1)

from keras.utils import to_categorical  # 위치값 잡아줌 -> 원핫
x1 = to_categorical(x)
y1 = to_categorical(y)
print(x1)
# [[[0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]
print(y1)
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]
print(x1.shape, y1.shape) # (12, 11) (9, 14)

#1. to_categorical에서 첫번째 0빼
x1 = x1[:,1:]
y1 = y1[:,1:]
# print(x1)
print(x1.shape, y1.shape) # (12, 10) (9, 13)


#2. 사이킷런 원핫인코더
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()   # sparse=False
x2_ohe = ohe.fit_transform(x).toarray() #배열로 바꿔준다.
y2_ohe = ohe.fit_transform(y).toarray() #배열로 바꿔준다.


# print(x2_ohe)

print(x2_ohe.shape) # (12, 8)
print(y2_ohe.shape) # (9, 6)


# #3. 판다스 겟더미
# x = np.array(x).reshape(12, )
# x3 = pd.get_dummies(x, dtype='int') # 겟더미는 백터만 받는다.
# print(x3)
# y = np.array(y).reshape(9, )
# y3 = pd.get_dummies(y, dtype='int')
# print(y3)

# print(x3.shape, y3.shape)   # (12, 8) (9, 6)


'''
'''