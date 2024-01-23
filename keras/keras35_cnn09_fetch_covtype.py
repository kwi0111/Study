from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(datasets)
# print(datasets.feature_names)
print(x.shape, y.shape) # (581012, 54) (581012,)
print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747
# Name: count, dtype: int64
print(y.shape)  # (581012,)

#1. 원핫엔코딩 keras
# from keras.utils import to_categorical    # 라벨값을 슬라이싱해서 0을 잘라내거나 밀어내거나 7을 0으로 바꾸거나
# y_ohe = to_categorical(y)
# print(y_ohe)
# print(y_ohe.shape)  #   (581012, 8)

# #2. 원핫엔코딩 pandas
# y_ohe = pd.get_dummies(y, dtype='int')
# print(y_ohe)    # [581012 rows x 7 columns]
# print(y_ohe.shape)    # (581012, 7)

#3. 원핫엔코딩 scikit-learn
y = y.reshape(-1, 1)
print(y.shape)  # (581012, 1)
ohe = OneHotEncoder(sparse=True)
y = ohe.fit_transform(y).toarray()
print(y)
print(y.shape)  # (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    train_size=0.9,
                                                    random_state=123,
                                                    stratify=y,
                                                    shuffle=True
                                                    ) 
# Label Set인 Y가 25%의 0과 75%의 1로 이루어진 Binary Set일 때, stratify=Y로 설정하면 나누어진 데이터셋들도 0과 1을 각각 25%, 75%로 유지한 채 분할된다.
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# scaler = MinMaxScaler() # 클래스 정의
scaler = StandardScaler() # 클래스 정의
# # scaler = MaxAbsScaler() # 클래스 정의
# # scaler = RobustScaler() # 클래스 정의

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train = x_train.reshape(-1, 9, 3, 2)
x_test = x_test.reshape(-1, 9, 3, 2)

#2. 모델구성
model = Sequential()
model.add(Conv2D(10, (1,1), input_shape = (9,3,2), activation='relu' ))
model.add(Flatten())
model.add(Dense(7, activation='softmax'))  

#3.컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc']
              )

es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=200,
                   verbose=1,
                   restore_best_weights=True
                   )
# mcp = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       verbose=1,
#                       save_best_only=True,
#                       filepath='../_data/_save/MCP/keras26_fetch_covtype_MCP1.hdf5'
#                       )
import time
start_time = time.time()   #현재 시간
hist = model.fit(x_train,
                 y_train,
                 epochs=10,
                 batch_size=2000,
                 validation_split=0.2,
                 callbacks=[es]
                 )
end_time = time.time()   #끝나는 시간

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict)
print(y_test)
print(y_predict.shape, y_test.shape) 

y_test = np.argmax(y_test, axis=1)
print(y_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
print(y_predict.shape, y_test.shape)   

acc = accuracy_score(y_test, y_predict)
print("로스 : ", results[0])
print("acc : ", results[1])
print("accuracy_score : ", acc)
print("걸린시간 : ", round(end_time - start_time, 2),"초")

# 씨피유
# 로스 :  0.6284416317939758
# acc :  0.7238649129867554
# accuracy_score :  0.7238649271969984
# 걸린시간 :  1799.05 초
