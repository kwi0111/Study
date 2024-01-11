import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping


#1. 데이터 

#1. 데이터 // 판다스, 넘파이 
path = "C:\\_data\\kaggle\\cancer\\"

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
print(x_train.shape, x_test.shape)      # (586, 7) (66, 7)   아웃컴, 인슐린 두개뺌
print(y_train.shape, y_test.shape)      # (586,) (66,)       train_size 달라지면 바뀜

print(x.shape, y.shape)     # (652, 7) (652,)



print(np.unique(y, return_counts=True))  # (array([0, 1], dtype=int64), array([424, 228], dtype=int64)), 라벨의 종류 별로 찾아줌 // 0과 1로 나눠져있다. // 0이 무엇인지 1이 무엇인지
print(pd.DataFrame(y). value_counts())
print(pd.Series(y).value_counts())
print(pd.value_counts(y))       # 행렬 데이터 일때 // mse로는 0과 1을 찾을수 없다. // 분류 // 



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


#2. 모델 구성
model = Sequential()
model.add(Dense(20, input_dim = 7)) 
model.add(Dense(80))
model.add(Dense(150))
model.add(Dense(850))
model.add(Dense(360))
model.add(Dense(80))
model.add(Dense(20))
model.add(Dense(1, activation='sigmoid')) # sigmoid 최종 레이어에 판단, 중간 레이어에 다 쓸수는 있음 // 렐루는 마지막에 잘안씀 0.5이상은 1, 0.5미만은 0으로 판단

#.3 컴파일 훈련 // 하나 포기하는게 과적합 안걸림 // 
model.compile(loss='binary_crossentropy', optimizer='adam',      # 이진 분류 'binary_crossentropy', 'sigmoid'// 공식중 두개가 공존하지 않음 // 실제값과 예측값의 차이
              metrics=['accuracy']                 # 0인지 1인지를 맞추는 정확성 // acc = accuracy // 'mse', 'mae' 도 넣을수 있음 
              )    
es = EarlyStopping(monitor = 'val_loss',
                   mode='auto',
                   patience=15,
                   verbose=1,
                   restore_best_weights=True 
                   )
hist = model.fit(x_train, y_train, epochs=100, batch_size=13, 
          validation_split=0.3, 
          callbacks=[es],    
          )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)   # acc에 필요함
y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)     # submit으로 제출해야하기 때문에,,


from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
r2 = r2_score(y_test, y_predict)    # 실제값, 예측값
print(y_test)
print(y_predict)        # 시그모이드 때문에 정수값이 아니다.




# ACC :  0.8333333333333334
def ACC(aaa, bbb):
    return accuracy_score(y_test, y_predict)
acc = ACC(y_test, y_predict)


######### submission.csv 만들기 (count칼럼에 값만 넣어주면 됨) #############
submission_csv['Outcome'] = y_submit
print(submission_csv)
print(submission_csv.shape)     # (116, 3)

submission_csv.to_csv(path + "submission_0111.csv", index=False) 

print("정확도 : " , acc)
print("로스, 정확도 : ", loss)
# print("r2 스코어 : " , r2)  # r2 조금 못미더움 // 정확도는 predict에 대한 결과 // 여기서는 필요없다.
# ACC :  0.8876253645985945
# 로스, 정확도 :  [0.4321504831314087, 0.7878788113594055]
# 무조건 val이 낫다.


# ACC :  0.9128709291752769
# 로스, 정확도 :  [0.4189327359199524, 0.8333333134651184]
# ACC :  0.9211323729436766
# 로스, 정확도 :  [0.4169829487800598, 0.8484848737716675]


