import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_iris()
print(datasets)     # 0 1 2 카테고리 크로스 엔트로피
print(datasets.DESCR)   # 라벨 = 클래스
print(datasets.feature_names)   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = datasets.data
y = datasets.target
print(x.shape, y.shape)         # (150, 4) (150,) 회귀데이터 분류 데이터 헷갈릴수 있음 // 라벨 개수 한쪽으로 쏠리면 과적합 발생할수 있음
print(y)
print(np.unique(y, return_counts=True))     # (array([0, 1, 2]), array([50, 50, 50], dtype=int64))
print(pd.value_counts(y))           # y라벨 클래스 개수 확인 
# 0    50
# 1    50
# 2    50
'''
# 원핫1 keras
from keras.utils import to_categorical
y_ohe = to_categorical(y)
print(y_ohe)
print(y_ohe.shape)   # (150, 3)


# 원핫2 판다스
y_ohe2 = pd.get_dummies(y, dtype = "int")
print(y_ohe2)
print(y_ohe2.shape)  # (150, 3)

'''
print('=========원핫3=============')
# 원핫3. 사이킷런
from sklearn.preprocessing import OneHotEncoder # 임포트 -> 정의 -> 핏 -> 트렌스폼
# y_ohe3 = y.reshape(-3,1)    # (150, 1)
# y = y.reshape(-1,1) # 백터를 행렬로 바꾼거
# y_ohe3 = y.reshape(150,1)   # (150, 1)
# y = y.reshape(50, 3)
# print(y.shape)
# print(y_ohe3)
# print(y_ohe3.shape)         # (150, 1)
# print(y.shape)
# ohe = OneHotEncoder(sparse=False)   # 디폴트 : true

ohe = OneHotEncoder(sparse=True)

# ohe.fit(y)    # 훈련
# y_ohe3 = ohe.transform(y)   # 트렌스폼에서 적용
# y_ohe3 = ohe.fit_transform(y)   # fit + transform 대신 쓴다.
y_ohe3 = ohe.fit_transform(y).toarray()   # fit + transform 대신 쓴다.
print(y_ohe3)
print(y_ohe3.shape)             # # (150, 3)

y = y_ohe3

x_train, x_test, y_train, y_test = train_test_split(
    x, y, # y를 y_ohe3 고쳐도됨              
    train_size=0.8,
    random_state=123,     
    stratify=y,     # 에러 : 분류에서만 쓴다. // y값이 정수로 딱 떨어지는것만 쓴다.
    shuffle=True,
    )
print(y_test)
print(np.unique(y_test, return_counts=True))        # (array([0, 1, 2]), array([10, 10, 10], dtype=int64))

#2. 모델 구성 
model = Sequential()
model.add(Dense(10, input_dim=4))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

#.3 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=30,
                   verbose=1,
                   restore_best_weights=True
                   )

hist = model.fit(x_train, y_train, epochs=10, batch_size=1,
                 validation_split=0.2,
                 callbacks=[es]
                 )

#4. 평가, 예측
results = model.evaluate(x_test, y_test)   
print("로스 : ", results[0])  # 로스 :  [0.17298726737499237, 0.8999999761581421] [로스 , ] : 로스 다음은 매트릭스에서 첫번째꺼
print("acc : ", results[1])  
y_predict = model.predict(x_test)
print(y_predict)
print(y_test)
print(y_predict.shape)  # (30, 3) 리스트 형식
print(y_test.shape) # (30, 3) 

y_test = np.argmax(y_test, axis=1)      # 열 중에 높은놈 잡아줘야함 // 원핫인코딩은 엑시스 1로 고정
y_predict = np.argmax(y_predict, axis=1)      # 최고값을 뽑아줘야함
print(y_test)
print(y_predict)
print(y_test.shape, y_predict.shape)


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : " , acc)





'''
print(y_test)
print(y_predict)

def ACC(a, b):
    return accuracy_score(a, b)
acc = ACC(y_test, y_predict)
print(acc)

print("정확도 : ", acc)

'''


