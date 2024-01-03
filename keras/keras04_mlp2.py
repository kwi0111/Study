# [실습]
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터 
x = np.array([[1,2,3,4,5,6,7,8,9,10],
            [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],   
            [9,8,7,6,5,4,3,2,1,0,],      # 파이썬에서는 끝에 ,있어도 에러 X
             ]
             )

y = np.array([1,2,3,4,5,6,7,8,9,10])
print(x)
print(x.shape, y.shape)     # (2, 10) (10,)
x = x.T
print(x.shape)  # (10, 2) 행무시 열우선

#2. 모델 구성
model = Sequential()
model.add(Dense(20, input_dim=3))   # input_dim = 3  입력 노드 3개 / 20 : 출력 노드 20개 
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=2)

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[10, 1.3, 0]])
print("로스 : ", loss)
print("[10, 1.3, 0]의 예측값 : ", results)


# 로스 :  7.376021130767185e-06
# [10, 1.3, 0]의 예측값 :  [[10.003944]]








