# 보스턴에 관한 데이터
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import time
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score


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

print(datasets.DESCR)   # 설명하다. 묘사하다. 데이터셋의 내용



x_train, x_test, y_train, y_test = train_test_split(x, y,               
                                                    train_size=0.7,
                                                    random_state=123,     
                                                    shuffle=True,
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
start_time = time.time()   #현재 시간
model.fit(x_train,y_train, epochs=100, batch_size=30, 
          validation_split = 0.3,   # 트레인에서 0.3개로 자른다. // 랜덤으로
          verbose=1
          )
end_time = time.time()   #끝나는 시간

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)   # 평가는 항상 테스트 데이터
y_predict = model.predict(x_test)
results = model.predict(x)

r2 = r2_score(y_test, y_predict)    # 실제값, 예측값
print("로스 : ", loss)
print("r2 스코어 : " , r2)
print("걸린시간 : ", round(end_time - start_time, 2),"초")

# 로스 :  54.067169189453125
# r2 스코어 :  0.3310843898530871
# 걸린시간 :  1.23 초

# validation_split 했을때
# 로스 :  43.44144058227539
# r2 스코어 :  0.46254524471575653
# 걸린시간 :  2.84 초


