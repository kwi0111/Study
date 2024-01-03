import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터 
x = np.array([[1,2,3,4,5,6,7,8,9,10],
            [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]       # 대괄호 -> 데이터 2개
             )

y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape)  #(2, 10)
print(y.shape)  #(10, )

# 전치행렬 : x = x.T으로 해도 된다.
x = x.transpose()
# [[1,1], [2, 1.1], [3, 1.2], ... [10, 1.3]]
print(x.shape)  #(10, 2)

#2. 모델 구성
model = Sequential()
model.add(Dense(20, input_dim=2))
#열, 컬럼, 속성, 특성, 차원 개수 = 2 // 같다.
# (행무시, 열우선) <= 암기
model.add(Dense(400))
model.add(Dense(1000))
model.add(Dense(300))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x,y, epochs=100, batch_size=2)

#4. 평가, 예측
loss = model.evaluate(x, y)
results = model.predict([[10, 1.3]])    # 스칼라, 벡터 개념 // 행의 개수 의미 X, 열 우선시 
print("로스 : ", loss)
print("[10, 1.3]의 예측값", results)

# [실습] : 10의 소수점 2째자리
# 로스 :  5.24365168530494e-06
# [10, 1.3]의 예측값 [[10.000706]]