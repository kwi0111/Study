# DNN으로 구성

# 48_2 카피
# (N ,4, 1) -> (N, 2, 2)
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


a = np.array(range(1,101))
x_predict = np.array(range(96,106)) # 
size = 5    # x데이터는 4개, y데이터는 1개

print(x_predict.shape)  # (10,)

def split_x(dataset, size):     
    aaa = []
    for i in range(len(dataset) - size + 1):
        # subset = dataset[i : (i + size)]
        # aaa.append(subset)
        aaa.append(dataset[i:i+size])
    return np.array(aaa)

bbb = split_x(a, size)      # a로 bbb를 만듦.
print(bbb)
print(bbb.shape)    #(96, 5)

ccc = split_x(x_predict, 4)
print(ccc)
print(ccc.shape)    #(7, 4)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape) # (96, 4) (96,)

x_predict = ccc
print(x_predict.shape) # (7, 4)


# 모델 구성 및 평가
# x = x.reshape(-1,2,2)
print(x.shape, y.shape) # (96, 4) (96,)
#2. 모델구성
model = Sequential()
model.add(Dense(100, input_shape = (4,), activation='sigmoid'))  #  행무시 열우선
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(10))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

# model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000)

#4. 평가, 예측
results = model.evaluate(x, y)
print("loss : ", results)

x_predict = np.array(x_predict)
print(x_predict.shape)  # 

y_pred = model.predict(x_predict)
print('[96, 106]의  결과 : ', y_pred)

'''
loss :  0.455750972032547
(7, 4)
1/1 [==============================] - 0s 45ms/step
[96, 106]의  결과 :  [[ 97.93484]
 [ 98.77517]
 [ 99.60858]
 [100.435  ]
 [101.25437]
 [102.06669]
 [102.87191]]
'''










