import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_diabetes

#1. 데이터
datasets = load_diabetes()
x = datasets.data       # 학습해야 할 feed용 데이터
y = datasets.target     # label 데이터, 예측해야 할 (class) 데이터

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
model.fit(x_train, y_train, epochs=1000, batch_size=2, 
          validation_split=0.5,
          verbose=0
          )   
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                                           
y_predict = model.predict(x_test)
# results = model.predict(x)
print(y_predict.shape)      # (54, 1)
# print(results.shape)        # (442, 1)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)                                               
print("로스 : ", loss)
print("r2 스코어 : " , r2)
print("걸린시간 : ", round(end_time - start_time, 2),"초")    




# validation_split=0.5
# 로스 :  38.55118942260742
# r2 스코어 :  0.6500460047192199
# 걸린시간 :  91.04 초

