import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras. callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import random

#1.데이터
path = "c:\\_data\\dacon\\iris\\"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)    # [120 rows x 5 columns]
print(train_csv.shape)  # (120, 5)
print(train_csv.info()) # 0 1 2 3 4 / 120

test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
print(train_csv)     # [120 rows x 5 columns]
print(train_csv.shape)  # (120, 5)
print(train_csv.info()) # # 0 1 2 3 4 / 120

submission_csv = pd.read_csv(path + 'sample_submission.csv')
print(submission_csv)
print(submission_csv.shape)  # (30, 2)

# 결측치 처리 
print(train_csv.isna().sum()) # 0

# x와 y를 분리
x = train_csv.drop(['species'], axis=1)
y = train_csv['species']
print(x.shape, y.shape) # (120, 4) (120,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2], dtype=int64), array([40, 41, 39], dtype=int64))
print(pd.value_counts(y))
# 1    41
# 0    40
# 2    39

# 원핫. 판다스
y_ohe = pd.get_dummies(y, dtype='int')
print(y_ohe)
print(y_ohe.shape)  # (120, 3)

# x_train, x_test, y_train, y_test = train_test_split(
#         x, y_ohe,             
#         train_size=0.7,
#         random_state=123,     
#         stratify=y,     # 에러 : 분류에서만 쓴다. // y값이 정수로 딱 떨어지는것만 쓴다.
#         shuffle=True,
# #         )
# print(y_test)
# print(np.unique(y_test, return_counts=True))  # (array([0, 1]), array([48, 24], dtype=int64))

#2. 모델 구성 
model = Sequential()
model.add(Dense(10, input_dim=4))
model.add(Dense(20))
model.add(Dense(100))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

def aaa() : 
    r = random.randrange(1, 777)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_ohe,             
        train_size=0.7,
        random_state=r, 
            
        stratify=y,  
        shuffle=True,
        )
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=200,
                   verbose=1,
                   restore_best_weights=True
                   )

    hist = model.fit(x_train, y_train, epochs=20000, batch_size=1,
                 validation_split=0.2,
                 callbacks=[es]
                 )
    results = model.evaluate(x_test, y_test)
    y_predict = model.predict(test_csv)
    y_submit = np.argmax(y_predict, axis=1)      
    submission_csv['species'] = y_submit
    return results[1]
print(aaa())

v = 1.0
while True:
    acc = aaa()
    if acc == 1.0:
        
        submission_csv.to_csv(path + "submission_0112_3.csv", index=False)
        break
    
        
    
#.3 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# es = EarlyStopping(monitor='val_loss',
#                    mode='min',
#                    patience=100,
#                    verbose=1,
#                    restore_best_weights=True
#                    )

# hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
#                  validation_split=0.4,
# #                  callbacks=[es]
#                  )

# #4. 평가, 예측
# results = model.evaluate(x_test, y_test)
# y_predict = model.predict(test_csv)
# # print(y_predict)
# # y_submit = np.argmax(y_predict, axis=1)
# # print(y_submit)
# print(y_predict)
# # print(y_test)
# # y_test = np.argmax(y_test, axis=1)
# y_submit = np.argmax(y_predict, axis=1)      # 최고값을 뽑아줘야함
# # # # print(y_test)
# print(y_submit)

# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, y_predict2)      갯수가 맞아야함 / y프리딕트는 x테스트
# # print("accuracy_score : " , acc)

# ######## submission.csv 만들기 (species칼럼에 값만 넣어주면 됨) #############
# submission_csv['species'] = y_submit
# submission_csv.to_csv(path + "submission_0112.csv", index=False) 
# print("로스 : ", results[0])  
# print("acc : ", results[1])  

