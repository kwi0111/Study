import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_diabetes

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) # (442, 10) (442,)
print(datasets.feature_names)   # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.88,
                                                    random_state=123,
                                                    shuffle=True
                                                    )

#2. 모델 구성
model = Sequential()
model.add(Dense(20, input_dim = 10))
model.add(Dense(30)) 
model.add(Dense(40)) 
model.add(Dense(20)) 
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mae", optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=2)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                                           
y_predict = model.predict(x_test)
results = model.predict(x)


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)                                               
print("로스 : ", loss)
print("r2 스코어 : " , r2)
print("걸린시간 : ", round(end_time - start_time, 2),"초")    

# mse -> mae
# 로스 :  45.504764556884766
# r2 스코어 :  0.48564100652133846
# 걸린시간 :  2.1 초

# 에포 1000
# 로스 :  45.126956939697266
# r2 스코어 :  0.4912459397013874
# 걸린시간 :  16.53 초

# 랜덤 스테이츠 123123 -> 123
# 로스 :  44.0648193359375
# r2 스코어 :  0.5049547290954755
# 걸린시간 :  5.26 초

# 에포 100 -> 50
# 로스 :  43.77748489379883
# r2 스코어 :  0.510466857146636
# 걸린시간 :  2.87 초

# 트레인 사이즈 0.7 -> 0.75
# 로스 :  43.56166076660156
# r2 스코어 :  0.5160067991116633
# 걸린시간 :  3.15 초

# 트레인 사이즈 0.75 -> 0.88
# 로스 :  39.65202713012695
# r2 스코어 :  0.6511779737698389
# 걸린시간 :  92.56 초


# R2 0.62 이상