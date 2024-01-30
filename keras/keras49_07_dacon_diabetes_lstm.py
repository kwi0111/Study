import time
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터 

#1. 데이터 // 판다스, 넘파이 
path = "C:\\_data\\dacon\\diabetes\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)    
print(train_csv)                                            # 여기 train_csv에서 훈련/테스트 데이터 나눠야함.
test_csv = pd.read_csv(path + "test.csv", index_col=0)      # 헤더는 기본 첫번째 줄이 디폴트값
print(test_csv)                                             # 위의 훈련 데이터로 예측해서 count값 찾아야함. (문제집)
submission_csv = pd.read_csv(path + "sample_submission.csv")       
print(submission_csv)                                       # 서브미션 형식 그대로 제출해야함.

# 3개의 파일을 메모리에 땡겨왔다. // 판다스 : 행과 열로 구성이 되어있는 DataFrame으로 읽는다 // 인덱스, 헤더 포함하는데 데이터로 안봄 /// 넘파이는 연산 

print(train_csv.shape)      # (652, 9)
print(test_csv.shape)       # (116, 8)
print(submission_csv.shape)    # (116, 2)
print(train_csv.columns)
# Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],


print(train_csv.info())         # 결측치 X
print(test_csv.info())          # 결측치 X

############# x 와 y를 분리 ################
x = train_csv.drop(['Outcome', 'Insulin'], axis=1)   # 행삭제 : axis = 0 // 열삭제 : axis = 1 // train_csv에 있는 'Outcome'열 삭제
print(x)
y = train_csv.drop(['Insulin'], axis=1)       # 행삭제 : axis = 0 // 열삭제 : axis = 1 // train_csv에 있는 'Outcome'열 삭제 
y = train_csv['Outcome']                      # train_csv에 있는 'Outcome'열을 y로 설정
test_csv = test_csv.drop(['Insulin'], axis=1)
print(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.9, random_state=123123,
)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# scaler = MinMaxScaler() # 클래스 정의
# scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
scaler = RobustScaler() # 클래스 정의

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
test_csv = scaler.fit_transform(test_csv)

x_train = x_train.reshape(-1, 7, 1)
x_test = x_test.reshape(-1, 7, 1)
test_csv = test_csv.reshape(-1, 7, 1)


print(x_train.shape, x_test.shape)      # (586, 7) (66, 7)   아웃컴, 인슐린 두개뺌
print(y_train.shape, y_test.shape)      # (586,) (66,)       train_size 달라지면 바뀜
print(x.shape, y.shape)     # (652, 7) (652,)

print(np.unique(y, return_counts=True))  # (array([0, 1], dtype=int64), array([424, 228], dtype=int64)), 라벨의 종류 별로 찾아줌 // 0과 1로 나눠져있다. // 0이 무엇인지 1이 무엇인지
print(pd.DataFrame(y). value_counts())
print(pd.Series(y).value_counts())
print(pd.value_counts(y))       # 행렬 데이터 일때 // mse로는 0과 1을 찾을수 없다. // 분류 // 



#2. 모델 구성
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape = (7,1))) 
model.add(LSTM(30))    
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid')) 


#.3 컴파일 훈련 // 하나 포기하는게 과적합 안걸림 // 
model.compile(loss='binary_crossentropy', optimizer='adam',      # 이진 분류 'binary_crossentropy', 'sigmoid'// 공식중 두개가 공존하지 않음 // 실제값과 예측값의 차이
              metrics=['accuracy']                 # 0인지 1인지를 맞추는 정확성 // acc = accuracy // 'mse', 'mae' 도 넣을수 있음 
              )    
es = EarlyStopping(monitor = 'val_loss',
                   mode='auto',
                   patience=150,
                   verbose=1,
                   restore_best_weights=True 
                   )
# mcp = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       verbose=1,
#                       save_best_only=True,
#                       filepath='../_data/_save/MCP/keras26_dacon_diabetes_MCP1.hdf5'
#                       )

hist = model.fit(x_train, y_train, epochs=1000, batch_size=13, 
          validation_split=0.2, 
          callbacks=[es],    
          )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)   # acc에 필요함
y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)     # submit으로 제출해야하기 때문에,,


# from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
# r2 = r2_score(y_test, y_predict)    # 실제값, 예측값
# print(y_test)
# print(y_predict)        # 시그모이드 때문에 정수값이 아니다.

# def ACC(aaa, bbb):
#     return accuracy_score(y_test, y_predict)
# acc = ACC(y_test, y_predict)


# ######### submission.csv 만들기 (count칼럼에 값만 넣어주면 됨) #############
# submission_csv['Outcome'] = y_submit
# print(submission_csv)
# print(submission_csv.shape)     # (116, 3)

# submission_csv.to_csv(path + "submission_0111.csv", index=False) 

# print("정확도 : " , acc)
print("로스, 정확도 : ", loss)
# print("걸린시간 : ", round(end_time - start_time, 2),"초")




# 씨피유
# 정확도 :  0.803030303030303
# 로스, 정확도 :  [0.42715394496917725, 0.8030303120613098]
# 걸린시간 :  66.18 초


# LSTM
# 로스, 정확도 :  [0.5873111486434937, 0.6969696879386902]