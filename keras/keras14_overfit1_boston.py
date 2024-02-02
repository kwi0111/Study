# 09_1 카피
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np

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
start_time = time.time()   #현재 시간

hist = model.fit(x_train, y_train, epochs=7, batch_size=1,       # 모델.핏에서 히스토리를 반환 // 히스토리에 반환값 있음.
          validation_split=0.2  # train_size=0.7에서 0.2
          ) 

end_time = time.time()   #끝나는 시간

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

print("걸린시간 : ", round(end_time - start_time, 2),"초")

print("==========================")
print(hist)
print("============= hist.history =============")
print(hist.history)         # 딕셔너리 {} : 키(로스,loss), 벨류(숫자,값) 한쌍 //
                            # 리스트 []: 두개이상
print("============ loss ============")
print(hist.history['loss'])
print("=========== val_loss ==========")
print(hist.history['val_loss'])
print("===============================")

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] ='Malgun Gothic'    # 위치
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss', marker='.')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')
# plt.legend(loc='upper right') # 라벨
# plt.title('보스턴 LOSS') #제목
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()
# plt.show()





# 오늘 과제 : 리스트, 닥셔너리, 튜플 공부
