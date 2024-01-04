import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

# [검색] train과 test를 섞어서 7:3으로 자를 수 있는 방법을 찾으시오.
# 힌트 : 사이킷런 // 랜덤하게 듬성듬성 이빨이 빠져야한다.

from sklearn.model_selection import train_test_split    # 네이밍룰   // 함수를 임포트 // sklearn pip때 설치 O

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,     # 디폴트 : 0.75
                                                    test_size=0.3,      # 디폴트 : 0.25  // 1이상 오버되면 에러 // 이하는 작동하지만 데이터 손실 O
                                                    random_state=1,     # 이것만 잘 조절해도 성적이 좋을 수 있다. // 데이터 클수록 발휘한다.
                                                    shuffle=True,       # 디폴트 : True  // False : 섞지 않겠다. 순차적으로
                                                    # stratify=         # 층을 이루게 하다, 계층화하다
                                                    )     
# 데이터 전처리가 끝났다. // 가중치 모델 더 신뢰 // 

# random_state : 안넣으면 계속 랜덤값으로 나옴 // 나오는 데이터 고정하기 위함. 그냥 무슨값 넣어도 상관 x // 좋은 데이터 찾았을때 그걸로 고정 // 랜덤 스테이츠 = 랜덤 난수표로 하면 데이터 신뢰  
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
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([x])
print("로스 : ", loss)
print("[11]의 예측값", results)


# 그림 그리는 API 땡겨오기
import matplotlib.pyplot as plt

plt.scatter(x, y)   # 흩뿌리다. 
plt.plot(x, results, color='red')
plt.show()



