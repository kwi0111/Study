import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 18], dtype=int64))
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
print(y)
print('---------------------------------------------------------')

x = x[:-35]
y = y[:-35]      # # 0을 줄여버리겠다.
print(x)
print(y)
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    13

# 불균형하게 데이터 만들어놨다 // 증폭하기 위해서

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=123,
    stratify=y
)

from keras.models import Sequential
from keras.layers import Dense
 
'''
#2.모델
model = Sequential()
model.add(Dense(128, input_shape = (13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))



#3.컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])    
# sparse_categorical_crossentropy 정수 형태의 타겟 값과 원-핫 인코딩된 타겟 값 둘 다를 지원
es = EarlyStopping(monitor='val_loss',
                   mode='auto,',
                   verbose=1,
                   restore_best_weights=True,
                   patience=10,
                   )
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4.평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print(y_test)       # 원핫 돼어있어
print(y_predict)    # 원핫 안되어있어 
y_predict = np.argmax(y_predict, axis=1)    # axis = 0 이면 행이라서 3개나옴
print(y_predict)
acc = accuracy_score(y_test, y_predict)
# print("정확도 : ", acc)
print('로스 : ', results[0])
print('정확도 : ', results[1])
f1 = f1_score(y_test, y_predict, average='macro')   # acc보다 조금더 정확 // 원래 2진분류에 쓰임 // average는 다중
print("f1 스코어 : ", f1)


# 지표 : f1_score
# f1_score는 양성 클래스와 음성 클래스의 불균형한 데이터셋에서 모델의 성능을 정확하게 측정할 수 있습니다. 
# 특히, 긍정 클래스와 부정 클래스의 비율이 크게 다를 때 사용하기 좋은 지표입니다.




# 로스 :  0.45682716369628906
# 정확도 :  0.9333333373069763
# f1 스코어 :  0.6470588235294118
'''

########################## smote ############################### 데이터 증폭하는데 좋음
# array([59, 71, 13]을 임의로 증폭
print("====================== smote 적용 =====================")
from imblearn.over_sampling import SMOTE
import sklearn as sk
print('사이킷런 : ', sk.__version__)    # 사이킷런 :  1.3.0

smote = SMOTE(random_state=123) # 랜덤 고정
x_train, y_train = smote.fit_resample(x_train, y_train) # 트레인 0.9 테스트 // 0.1은 그대로 (평가는 증폭 X)
print(pd.value_counts(y_train))
# 0    63
# 2    63
# 1    63

#2.모델
model = Sequential()
model.add(Dense(128, input_shape = (13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))



#3.컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])    
# sparse_categorical_crossentropy 정수 형태의 타겟 값과 원-핫 인코딩된 타겟 값 둘 다를 지원
es = EarlyStopping(monitor='val_loss',
                   mode='auto,',
                   verbose=1,
                   restore_best_weights=True,
                   patience=10,
                   )
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4.평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print(y_test)       # 원핫 돼어있어
print(y_predict)    # 원핫 안되어있어 
y_predict = np.argmax(y_predict, axis=1)    # axis = 0 이면 행이라서 3개나옴
print(y_predict)
acc = accuracy_score(y_test, y_predict)
# print("정확도 : ", acc)
print('로스 : ', results[0])
print('정확도 : ', results[1])
f1 = f1_score(y_test, y_predict, average='macro')   # acc보다 조금더 정확 // 원래 2진분류에 쓰임 // average는 다중
print("f1 스코어 : ", f1)

# 로스 :  0.6314700841903687
# 정확도 :  0.7333333492279053
# f1 스코어 :  0.5047619047619047





