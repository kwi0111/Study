import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터 

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y,               # x, -> 엑스트레인, 엑스 테스트  y, -> 와이 트레인, 와이 테스트 순서 // 순서 바꾸면 이상한 값
                                                    test_size=0.7,
                                                    random_state=12,     # 랜덤 난수
                                                    shuffle=True,
                                                    )
 
print(x_train)
print(y_train)
print(x_test)
print(y_test)

#2. 모델 구성 
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(80))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, epochs=10000, batch_size=1)

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

# [실습]
# r2 : 0.87이상

# 로스 :  15.241659164428711
# r2 스코어 :  0.6236153186532674

# 에포 100 -> 50
# 로스 :  14.059799194335938
# r2 스코어 :  0.6528007192868279

# 에포 50 -> 10
# 로스 :  10.704954147338867
# r2 스코어 :  0.7356468454709838

# 배치 사이즈 1 -> 2
# 로스 :  10.114675521850586
# r2 스코어 :  0.7502234674642463

# 랜덤 난수만 수정
# 로스 :  6.154726982116699
# r2 스코어 :  0.8443248820971646









