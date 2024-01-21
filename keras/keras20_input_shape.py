# 09_1 가져온 데이터
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


from sklearn.datasets import load_boston
# 현재 사이킷런 버젼 1.3.0 보스턴 안됨. 그래서 삭제
# pip uninstall scikit-learn
# pip uninstall scikit-image        
# pip uninstall scikit-learn-intelex
# pip install scikit-learn==1.1.3

#1. 데이터
datasets = load_boston()        # 변수에 집어 넣은다음 프린트
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



x_train, x_test, y_train, y_test = train_test_split(x, y,               
                                                    train_size=0.7,
                                                    random_state=1140,     
                                                    shuffle=True,
                                                    )

#2. 모델 구성 
model = Sequential()
# model.add(Dense(5, input_dim=13))           
model.add(Dense(10, input_shape = (13, )))  # 백터 형태로 들어감. -> 행무시 열우선 (1000, 100, 100) 이면 (100, 100) // 원래 (nan, 13)
model.add(Dense(10))
model.add(Dense(40))
model.add(Dense(60))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss="mse", optimizer='adam')
start_time = time.time()   #현재 시간

model.fit(x_train, y_train, epochs=100, batch_size=1)
end_time = time.time()   #끝나는 시간

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)   # 평가는 항상 테스트 데이터
y_predict = model.predict(x_test)
results = model.predict(x)  # 전체 데이터셋 x에 대한 모델의 예측값이 들어가게 된다.

from sklearn.metrics import r2_score    #
r2 = r2_score(y_test, y_predict)    # 실제값, 예측값
print("로스 : ", loss)
print("r2 스코어 : " , r2)
print("걸린시간 : ", round(end_time - start_time, 2),"초")     # 









# [실습]
# train_size 0.7이상, 0.9이하
# R2 0.8 이상
# 로스 튀면 과적합

# 로스 :  29.739749908447266
# r2 스코어 :  0.6743844503014744

# 로스 :  23.210712432861328
# r2 스코어 :  0.7458697566068171

# 에포 : 10000
# 로스 :  22.44500732421875
# r2 스코어 :  0.7542533550262869
# 걸린시간 :  2769.34 초




