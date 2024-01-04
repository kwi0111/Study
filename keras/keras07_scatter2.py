import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터 

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y,               # x -> 엑스트레인, 엑스 테스트  y -> 와이 트레인, 와이 테스트 순서 // 순서 바꾸면 이상한 값
                                                    test_size=0.7,
                                                    random_state=9,     # 랜덤 난수
                                                    shuffle=True,
                                                    )
 
print(x_train)
print(y_train)
print(x_test)
print(y_test)

#2. 모델 구성 
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([x])
print("로스 : ", loss)
print("[11]의 예측값", results)

# 모델 구성후 그리기. -> 그래프 그린다.(시각화한다.)
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.scatter(x, results, c='r', marker = 'x')
# plt.plot(x, results, c='r')
plt.show()

# 파란선 :  실제 데이터 // 빨간선 : 예측 데이터





