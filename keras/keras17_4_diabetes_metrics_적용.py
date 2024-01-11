import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split


#1. 데이터
datasets = load_diabetes()
x = datasets.data       # 학습해야 할 feed용 데이터
y = datasets.target     # label 데이터, 예측해야 할 (class) 데이터

print(x)
print(y)
print(x.shape, y.shape) # (442, 10) (442,)
print(datasets.feature_names)   # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.88,
                                                    random_state=123,
                                                    shuffle=True
                                                    )

#2. 모델 구성
model = Sequential()
model.add(Dense(20, input_dim = 10, activation='relu'))
model.add(Dense(30, activation='relu')) 
model.add(Dense(40, activation='relu')) 
model.add(Dense(20, activation='relu')) 
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mae", optimizer='adam')
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=60,
                   verbose=1,
                   restore_best_weights=True
                   )

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=2, 
          validation_split=0.3,
          verbose=1,
          callbacks=[es]
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


def RMSE(aaa, bbb):
    return np.sqrt(mean_squared_error(aaa, bbb))
rmse = RMSE(y_test, y_predict)
print("RMSE : " , rmse)
print("MSE : ", loss)
print("r2 스코어 : " , r2)
print("걸린시간 : ", round(end_time - start_time, 2),"초")    

