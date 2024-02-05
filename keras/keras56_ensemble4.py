import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate, concatenate


#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T  # 삼성전자 종가, 하이닉스 종가


print(x1_datasets.shape)     # (100, 2) (100, 3)

y1 = np.array(range(3001, 3101)) # 비트코인 종가
y2 = np.array(range(13001, 13101)) # 이더리움 종가


x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets,  y1, y2, train_size=0.7, random_state=123,
)

print(x1_train.shape, y1_train.shape, y2_train.shape)     # (70, 2) (70, 3) (70,)

#2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='bit1')(input1)  # name 라벨링
dense2 = Dense(10, activation='relu', name='bit2')(dense1)
dense3 = Dense(10, activation='relu', name='bit3')(dense2)
output1 = Dense(10, activation='relu', name='bit4')(dense3) # 시작
# model = Model(inputs=input1, outputs=output1)
# model.summary()


#2-3. concatnate 사슬처럼 엮을게 없다.
merge1 = Dense(7, name='mg2')(output1)
merge2 = Dense(11, name='mg3')(merge1)
last_output1 = Dense(1, name='last1')(merge2) # 끝
last_output2 = Dense(1, name='last2')(merge2)

model = Model(inputs = input1, outputs = [last_output1, last_output2])

model.summary() # 전체 데이터 갯수 맞춰줘야한다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x1_train, [y1_train, y2_train], epochs=300, batch_size=2, verbose=1)

#4. 평가 및 예측
results = model.evaluate(x1_test, [y1_test, y2_test])
y_predict = model.predict(x1_test)
print("loss통합 : ", results[0])
print("x1_test, y1_test : ", results[1])
print("x2_test, y2_test : ", results[2])
# print(y_predict)
# print(len(x1_test))
# print(y)
# print(y_train)

