'''
고의적으로 R2값 낮추기
1. R2를 음수가 아닌 0.5이하로 만들것
2. 데이터는 건들지 말것
3. 레이어는 인풋과 아웃풋 포함하여 7개 이상
4. batch_size=1
5. 히든레이어의 노드는 10개 이상 100개 이하
6. train 사이즈 75%
7. epochs 100번 이상
'''


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터 

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y,               # x, -> 엑스트레인, 엑스 테스트  y, -> 와이 트레인, 와이 테스트 순서 // 순서 바꾸면 이상한 값
                                                    test_size=0.75,
                                                    random_state=12,     # 랜덤 난수
                                                    shuffle=True,
                                                    )
 
print(x_train)
print(y_train)
print(x_test)
print(y_test)

#2. 모델 구성 
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(10))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)   # 평가는 항상 테스트 데이터
y_predict = model.predict(x_test)
results = model.predict(x)

from sklearn.metrics import r2_score    #
r2 = r2_score(y_test, y_predict)    # 실제값, 예측값
print("로스 : ", loss)
print("r2 스코어 : " , r2)

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.scatter(x, results, c='r', marker = 'x')
plt.show()

# 
# 로스 :  21.568782806396484
# r2 스코어 :  0.43478029377671157



