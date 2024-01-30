import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM


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
print(bbb.shape)
# (96, 5)
ccc = split_x(x_predict, 4)
print(ccc)
print(ccc.shape)    #(6, 5)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape) # (96, 4) (96,)

x_predict = ccc
print(x_predict.shape) # (7, 4)


# 모델 구성 및 평가
x = x.reshape(-1,4,1)
print(x.shape, y.shape) # (96, 4, 1) (96,)

#2. 모델구성
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape = (4, 1), activation='sigmoid'))  #  2차원인데 밑에는 3차원을 받는다. 실행하려면 return_sequences=True
model.add(LSTM(30, return_sequences=True, activation='sigmoid'))    # 두개 이상이라고해서 좋아진다는 보장 x
model.add(LSTM(100, return_sequences=True, activation='sigmoid'))
model.add(LSTM(10, activation='sigmoid'))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(1))

# model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000)

#4. 평가, 예측
results = model.evaluate(x, y)
print("loss : ", results)

x_predict = np.array(x_predict).reshape(-1,4,1)
print(x_predict.shape)  # (7, 4, 1)

y_pred = model.predict(x_predict)
print('[96, 106]의  결과 : ', y_pred)



# loss :  0.0016706595197319984
# (7, 4, 1)
# 1/1 [==============================] - 0s 265ms/step
# [96, 106]의  결과 :  [[ 99.81917 ]
#  [100.591774]
#  [101.28828 ]
#  [101.9069  ]
#  [102.44923 ]
#  [102.91956 ]
#  [103.3241  ]]





