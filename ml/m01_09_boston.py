# 09_1에서 가져옴

# 보스턴에 관한 데이터
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import time
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVR


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
                                                    train_size=0.7,
                                                    random_state=1140,     
                                                    shuffle=True,
                                                    )


# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 0.0
# print(np.min(x_test))  # -0.010370370370370367
# print(np.max(x_train))  # 1.0000000000000002
# print(np.max(x_test))  # 1.0280851063829786

#2. 모델 구성 
model = LinearSVR(C=100)


#3. 컴파일, 훈련 
start_time = time.time()   #현재 시간
model.fit(x_train, y_train)
end_time = time.time()   #끝나는 시간

#4. 평가, 예측
results = model.score(x_test, y_test) 
y_predict = model.predict(x_test) 
print("acc : ", results)
print("걸린시간 : ", round(end_time - start_time, 2),"초")     # 

# acc :  0.5275130859946872
# 걸린시간 :  0.01 초

# 스케일러 x
# acc :  0.19233716312185622
# 걸린시간 :  0.01 초


# 그냥
# 로스 :  39.140785217285156

# MinMaxScaler
# 로스 :  24.994285583496094


# StandardScaler
# 로스 :  23.470680236816406


# MaxAbsScaler
# 로스 :  22.6903018951416

# RobustScaler
# 로스 :  24.349830627441406
