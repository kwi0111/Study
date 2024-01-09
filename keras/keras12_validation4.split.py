# fit에서 발리데이션

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

#1.데이터 
x = np.array(range(1, 17))      # 1 ~ 16
y = np.array(range(1, 17))



x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    shuffle=True,
                                                    random_state=123,
                                                    train_size = 0.85
                                                    )
print(x_train, y_train)
print(x_test, y_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(20, input_dim=1))
model.add(Dense(400))
model.add(Dense(1000))
model.add(Dense(300))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x_train,y_train, epochs=1000, batch_size=1, 
          validation_split = 0.3,   # 13개 트레인에서 0.3개로 자른다. // 랜덤으로
          verbose=1
          )


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([11000])     
print("로스 : ", loss)
print("[11000]의 예측값", results)




