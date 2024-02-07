import numpy as np
from sklearn.datasets import load_breast_cancer     # 유방암
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC


#1. 데이터 
datasets = load_breast_cancer()
# print(datasets)   # 0,1때 걸렷냐 안걸렷냐
# 010101 일때 회귀로 가능할까?
print(datasets.DESCR)   
print(datasets.feature_names)

x = datasets.data
y = datasets.target 
# print(x.shape, y.shape)     # (569, 30) (569,)

# print(np.unique(y, return_counts=True))       # (array([0, 1]), array([212, 357], dtype=int64)) ,라벨의 종류 별로 찾아줌 // 0과 1로 나눠져있다. // 0이 무엇인지 1이 무엇인지
# print(pd.DataFrame(y).value_counts())
# print(pd.Series(y).value_counts())
# print(pd.value_counts(y))                     # 행렬 데이터 일때 // mse로는 0과 1을 찾을수 없다. // 


# # 넘파이 0과 1의 갯수가 몇개인지 찾아라.
# unique, counts = np.unique(y, return_counts=True)
# print(unique, counts)    # [0 1] [212 357]
# print("고유한 요소:", unique)   # [0 1]
# print("각 요소의 개수:", counts)    # [212 357]

# 판다스 0과 1의 갯수가 몇개인지 찾아라.
# pd.unique(y)
# print(pd.unique(y))     # [0 1]
# unique_values = pd.unique(y)  # 고유한 값들을 추출
# print(unique_values, counts)    # [0 1] [212 357]

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    random_state=123,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    )
# from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
# from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# # scaler = MinMaxScaler() # 클래스 정의
# # scaler = StandardScaler() # 클래스 정의
# # scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의


# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)



#2. 모델 구성
model = LinearSVC(C=100)

# model = Sequential()
# model.add(Dense(10, input_dim = 30))        
# model.add(Dense(80))
# model.add(Dense(30))
# model.add(Dense(10))
# model.add(Dense(1, activation='sigmoid')) 

# sigmoid : 모든값 0~1로 한정, 0.5이상은 1, 0.5미만은 0으로 판단 // 회귀 모델은 보통 리니어(디폴트값) // 2진분류는 sigmoid에 binary_crossentropy
# sigmoid 최종 레이어에 쓴다 // 중간 레이어에 다 쓸수는 있음 // 렐루는 마지막에 잘안씀 

#.3 컴파일 훈련 // 하나 포기하는게 과적합 안걸림 // 
model.fit(x_train, y_train)

# model.compile(loss='binary_crossentropy',  # loss='binary_crossentropy' -> 이진 분류때 씀 (sigmoid) // 실제값과 예측값의 차이
#               optimizer='adam',      
#               metrics=['accuracy']         # 0인지 1인지를 맞추는 정확성 // acc = accuracy // 'mse', 'mae' 도 넣을수 있음
#               )    
#                 # 둘중 하나는 무조건 'binary_crossentropy
# es = EarlyStopping(monitor = 'val_loss',
#                    mode='min',
#                    patience=50,
#                    verbose=1,
#                    restore_best_weights=True
#                    )

# hist = model.fit(x_train, y_train, epochs=800, batch_size=12, 
#           validation_split=0.3, 
#           callbacks=[es]        
#           )

#4. 평가, 예측
loss = model.score(x_test, y_test) 
y_predict = model.predict(x_test)
y_predict = y_predict.round()

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
r2 = r2_score(y_test, y_predict)    # (실제값, 예측값)
print(y_test)
print(y_predict)        # 시그모이드 때문에 정수값이 아니기 때문에 반올림 처리


def ACC(aaa, bbb):
    return (accuracy_score(aaa, bbb))
acc = ACC(y_test, y_predict)

print("ACC : " , acc)
print("로스, 정확도 : ", loss)
print("r2 스코어 : " , r2)  # r2 조금 못미더움 // 정확도는 predict에 대한 결과

'''
# 로스, 정확도 :  [0.06485937535762787, 0.9824561476707458]
# r2 스코어 :  0.9397867375545318

'''
# 그냥
# 0.0695752426981926

# MinMaxScaler
# 0.11030033230781555

# StandardScaler
# 0.11294618248939514

# MaxAbsScaler
# 0.073776975274086

# RobustScaler
# 0.08058270066976547

#
# ACC :  0.956140350877193
# 로스, 정확도 :  0.956140350877193
# r2 스코어 :  0.8095556298028733
