from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

datasets = load_wine()
x = datasets.data
y = datasets.target
print(datasets) # 클래스 0 1 2 다중 분류
print(datasets.feature_names)
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
print(datasets.DESCR)
print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True))
print(pd.value_counts(y))       # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# 1    71
# 0    59
# 2    48

print(y.shape)  # (178,)
'''
#1. 원핫엔코딩 keras 
from keras.utils import to_categorical # 10진수를 2진수로 바꿔주는 함수
y_ohe = to_categorical(y)
print(y_ohe)
print(y_ohe.shape)  # (178, 3)
'''
'''
#2. 원핫엔코딩 판다스
y_ohe = pd.get_dummies(y, dtype='int')
print(y_ohe)
print(y_ohe.shape)  # (178, 3)
'''

#3. 원핫엔코딩 사이킷런
from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1, 1)    
#  차원을 재구조화 및 변경하고자 할 때 reshape() 함수를 사용
# 원래 y는 (178,) --> reshape로 (178, 1)
# - reshape(-1, 정수) 일때
# : 행 자리에 -1, 그리고 열 위치에 임의의 정수가 있을 때 정수에 따라서 178개의 원소가 해당 열 개수만큼 자동으로 구조화
print(y.shape)  # (178, 1)

ohe = OneHotEncoder(sparse=True)
y = ohe.fit_transform(y).toarray()
print(y)
print(y.shape)  # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    train_size=0.8,
                                                    random_state=123,
                                                    stratify=y,
                                                    shuffle=True
                                                    )
print(y_test)
print(np.unique(y_test, return_counts=True))    # (array([0., 1.]), array([72, 36], dtype=int64))

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# scaler = MinMaxScaler() # 클래스 정의
scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

#3.컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc']
              )

# es = EarlyStopping(monitor='val_loss',
#                    mode='auto',
#                    patience=25,
#                    verbose=1,
#                    restore_best_weights=True
#                    )
mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath='../_data/_save/MCP/keras26_wine_MCP1.hdf5'
                      )
import time
start_time = time.time()   #현재 시간
hist = model.fit(x_train,
                 y_train,
                 epochs=1000,
                 batch_size=10,
                 validation_split=0.3,
                 callbacks=[mcp]
                 )
end_time = time.time()   #끝나는 시간

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict)
print(y_test)
print(y_predict.shape, y_test.shape)    # (36, 3) (36, 3)

y_test = np.argmax(y_test, axis=1)
print(y_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
print(y_predict.shape, y_test.shape)    # (36,) (36,)

acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)
print("로스 : ", results[0])
print("acc : ", results[1])
print("걸린시간 : ", round(end_time - start_time, 2),"초")


















 




