# 14_1 카피
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston


#1. 데이터
datasets = load_boston()  
print(datasets)
x = datasets.data
y = datasets.target
print(x)
print(x.shape)  #(506, 13) 컬럼이 무엇인지 모름.
print(y)
print(y.shape)  #(506,)

print(datasets.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

print(datasets.DESCR)   #설명하다. 묘사하다. 데이터셋의 내용

x_train, x_test, y_train, y_test = train_test_split(
    x, y,               
    train_size=0.7,
    random_state=1140,     
    # shuffle=True,
    )

#2. 모델 구성 
model = Sequential()
model.add(Dense(5, input_dim=13))           # 행무시, 열우선
model.add(Dense(10))
model.add(Dense(40))
model.add(Dense(60))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss="mse", optimizer='adam')
es = EarlyStopping(monitor='val_loss',
                   mode='min',      # min, max, auto
                   patience=10,
                   verbose=1,
                   )

hist = model.fit(x_train, y_train, epochs=200, batch_size=1, 
          validation_split=0.2, 
          callbacks=[es]        
          )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)   # 평가는 항상 테스트 데이터
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)    # 실제값, 예측값
print("로스 : ", loss)
print("r2 스코어 : " , r2)

def RMSE(aaa, bbb):
    return np.sqrt(mean_squared_error(aaa, bbb))
rmse = RMSE(y_test, y_predict)
print("RMSE : " , rmse)
print("MSE : ", loss)











