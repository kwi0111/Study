from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
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
# scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의


# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)



print(y_test)
print(np.unique(y_test, return_counts=True))    # (array([0., 1.]), array([697218, 116203], dtype=int64))

#2. 모델구성
model = Sequential()
model.add(Dense(80, input_dim=54))
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(20))
model.add(Dense(7, activation='softmax'))

#3.컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc']
              )

es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=100,
                   verbose=1,
                   restore_best_weights=True
                   )
hist = model.fit(x_train,
                 y_train,
                 epochs=10000,
                 batch_size=1000,
                 validation_split=0.3,
                 callbacks=[es]
                 )

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("로스 : ", results[0])
print("acc : ", results[1])
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
print("accuracy_score : ", acc)

# accuracy_score :  0.7971825168024922

# 그냥
# 로스 :  1.098958134651184

# MinMaxScaler
# 로스 :  1.091755747795105

# StandardScaler
# 로스 :  1.092885971069336

# MaxAbsScaler
# 로스 :  1.0899087190628052

# RobustScaler
# 로스 :  1.09199857711792