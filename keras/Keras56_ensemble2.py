import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate, concatenate





#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T  # 삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).T # 원유, 환율, 금시세
x3_datasets = np.array([range(100), range(301, 401), range(77, 177),range(33, 133)]).T # 1 ,2,3,4

print(x1_datasets.shape, x2_datasets.shape)     # (100, 2) (100, 3)

y = np.array(range(3001, 3101)) # 비트코인 종가

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets,  y, train_size=0.7, random_state=123,
)

print(x1_train.shape, x2_train.shape, y_train.shape)     # (70, 2) (70, 3) (70,)

#2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='bit1')(input1)  # name 라벨링
dense2 = Dense(10, activation='relu', name='bit2')(dense1)
dense3 = Dense(10, activation='relu', name='bit3')(dense2)
output1 = Dense(10, activation='relu', name='bit4')(dense3) # 시작
# model = Model(inputs=input1, outputs=output1)
# model.summary()

#2-2. 모델
input11 = Input(shape=(3,))
dense11 = Dense(100, activation='relu', name='bit11')(input11)
dense12 = Dense(100, activation='relu', name='bit12')(dense11)
dense13 = Dense(100, activation='relu', name='bit13')(dense12)
output11 = Dense(5, activation='relu', name='bit14')(dense13)    # 시작
# model = Model(inputs=input11, outputs=output11)
# model.summary()

#2-3. 모델
input111 = Input(shape=(4,))
dense111 = Dense(10, activation='relu', name='bit111')(input111)
dense112 = Dense(10, activation='relu', name='bit112')(dense111)
dense113 = Dense(10, activation='relu', name='bit113')(dense112)
output111 = Dense(10, activation='relu', name='bit114')(dense113)





#2-4. concatnate 사슬처럼 엮다.
merge1 = concatenate([output1, output11, output111], name='mg1')
merge2 = Dense(7, name='mg2')(merge1)
merge3 = Dense(11, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3) # 끝

model = Model(inputs = [input1, input11, input111], outputs = [last_output])

model.summary() # 전체 데이터 갯수 맞춰줘야한다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], y_train, epochs=100, batch_size=2, verbose=1)

#4. 평가 및 예측
results = model.evaluate([x1_test, x2_test, x3_test], y_test)
y_predict = model.predict([x1_test, x2_test, x3_test])
print("loss : ", results)
# print(y_predict)
# print(len(x1_test))
# print(y)
# print(y_train)

