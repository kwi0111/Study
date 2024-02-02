# 09_1에서 가져옴

# 보스턴에 관한 데이터
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import time
from sklearn.datasets import load_boston

#1. 데이터
datasets = load_boston()        # 변수에 집어 넣은다음 프린트
print(datasets)
x = datasets.data           # x에서 스케일링
y = datasets.target         # y 건들지 않는다.
print(x)
print(x.shape)  #(506, 13) 컬럼이 무엇인지 모름.
print(y)
print(x.shape, y.shape)  #(506, 13), (506,)

print(datasets.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

print(datasets.DESCR)   #설명하다. 묘사하다. 데이터셋의 내용


# 스케일러 x_train 에서 해야함
x_train, x_test, y_train, y_test = train_test_split(x, y,               
                                                    train_size=0.8,
                                                    random_state=1140,     
                                                    shuffle=True,
                                                    )
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# scaler = MinMaxScaler() # 클래스 정의
scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의


x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#2. 모델 구성 
# model = Sequential()
# model.add(Dense(5, input_dim=13)) 
# model.add(Dense(10))
# model.add(Dense(40))
# model.add(Dense(60))
# model.add(Dense(20))
# model.add(Dense(1))

#2. 모델 구성 (함수형) // 동일한 모델이다.
input1 = Input(shape=(13,))
dense1 = Dense(5)(input1)  # 그 레이어(input1)의 인풋 레이어 
dense2 = Dense(10)(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(40, activation = 'relu')(dense2)
drop1 = Dropout(0.2)(dense3)
dense4 = Dense(10)(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs = input1, outputs = output1) 

#3. 컴파일, 훈련 
model.compile(loss="mse", optimizer='adam')

start_time = time.time()   #현재 시간
model.fit(x_train, y_train, epochs=100, batch_size=10)
end_time = time.time()   #끝나는 시간

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)   # 평가는 항상 테스트 데이터
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("로스 : ", loss)
print("r2 스코어 : " , r2)
print("걸린시간 : ", round(end_time - start_time, 2),"초")

