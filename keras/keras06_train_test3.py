import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_iris

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

# [검색] train과 test를 섞어서 7:3으로 자를 수 있는 방법을 찾으시오.
# 힌트 : 사이킷런 // 랜덤하게 듬성듬성 이빨이 빠져야한다.


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1, shuffle=True)     # train_size=0.7 == test_size=0.3 // 
# random_state : 안넣으면 계속 랜덤값으로 나옴 // 나오는 데이터 고정하기 위함. 그냥 무슨값 넣어도 상관 x // 좋은 데이터 찾았을때 그걸로 고정
# 셔플과 랜덤 스테이트
# 사이킷런 API 엄청난놈 //  train_test_split를 쓴다.


print(x_train)  # [5 1 4 2 8 9 6]
print(y_train)  # [6 1 4 2 8 9 5]
print(x_test)   # [ 3 10  7]
print(y_test)   # [ 3 10  7]

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([11000, 7])
print("로스 : ", loss)
print("[11000, 7]의 예측값", results)


