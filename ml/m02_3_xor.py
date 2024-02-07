# 논리 회로 연산 2진 연산
# and : 곱셈
# or : 덧셈
# xor : 
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

#1.데이터 (and)
x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0, 1, 1, 0])     # y값만 차이
print(x_data.shape, y_data.shape)   # (4, 2) (4,)

#2. 모델
model = LinearSVC()
# model = Perceptron()


#3. 훈련
model.fit(x_data, y_data)

#4. 평가, 예측
acc = model.score(x_data, y_data)
print('model.score : ', acc)

y_predict = model.predict(x_data)
acc2 = accuracy_score(y_data, y_predict)
print('model.score : ', acc2)

print('++++'*20)
print(y_data)
print(y_predict)
# [0 1 1 0]
# [0 0 0 0] 안맞는다.
