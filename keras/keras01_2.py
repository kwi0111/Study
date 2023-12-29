# from tensorflow.keras.models import Sequential
# from tensorflow.keras.laters import Dense            라인 주석 처리 : 컨트롤 + /
import numpy as np
from keras.models import Sequential
from keras.layers import Dense



#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))
            # 1=outputlayer, input layer, 신경망=hidden layer

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x, y, epochs =10000)  # 에포 튜닝을 해야한다.


#4. 평가, 예측
loss = model.evaluate(x, y)
print("로스 : ", loss)
result = model.predict([1,2,3,4,5,6,7])
print("7의 예측값 : ", result)


# 로스 :  0.3238096535205841
# 1/1 [==============================] - 0s 58ms/step
# 7의 예측값 :  [[1.1428585]
#  [2.085715 ]
#  [3.0285716]
#  [3.9714282]
#  [4.914285 ]
#  [5.8571415]
#  [6.799998 ]]
# 에포 : 10000