# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, mean_squared_error


#1. 데이터          // 판다스, 넘파이
path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 인덱스 칼럼 : 0번째 //  \, \\, /, // 다 된다.   // read_csv : csv 파일을 불러오기 위한 함수
print(train_csv)                                            # 여기 train_csv에서 훈련/테스트 데이터 나눠야함.
test_csv = pd.read_csv(path + "test.csv", index_col=0)      # 헤더는 기본 첫번째 줄이 디폴트값
print(test_csv)                                             # 위의 훈련 데이터로 예측해서 count값 찾아야함.
submission_csv = pd.read_csv(path + "submission.csv")       
print(submission_csv)                                       # 서브미션 형식 그대로 제출해야함.

# 3개의 파일을 메모리에 땡겨왔다. // 판다스 1차원 백터 시리즈 2차원 행렬 데이터 프레임 // 인덱스, 헤더 포함하는데 데이터로 안봄 /// 넘파이는 연산 

print(train_csv.shape)      #(1459, 10) // 원래 11개인데 위에서 첫번째 칼럼을 인덱스로 바꿈
print(test_csv.shape)       #(715, 9)   // 원래 10개 
print(submission_csv.shape)     #(715, 2)

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'], 
#       dtype='object')
print(train_csv.info())         # Non-Null Count : 결측치가 아닌 데이터 갯수

print(test_csv.info())          # 컬럼 이름, 결측을 제외한 값 카운트, 타입을 보여줌.

print(train_csv.describe())     # describe 함수 (괄호 있어야함) // 데이터 프레임 컬럼별 카운트, 평균, 표준편차, 최소값, 4분위 수, 최대값을 보여줌.




############## 결측치 처리, 1.제거 ############
# print(train_csv.isnull().sum())       
print(train_csv.isna().sum())           # 데이터 프레임 결측치 확인
train_csv = train_csv.dropna()          # 결측치있으면 행이 삭제됨
print(train_csv.isna().sum())           # 결측치 삭제 후 확인
print(train_csv.info())
print(train_csv.shape)                  # (1328, 10)        // 1459 - 1328 = 121열 삭제


############## 결측치 처리, 2.채움 ############
test_csv = test_csv.fillna(test_csv.mean())     # train에서 없어진 결측치를 평균인 중간값으로 채움.
print(test_csv.info())


############ x 와 y를 분리 ################
x = train_csv.drop(['count'], axis=1)       # 행삭제 : axis = 0 // 열삭제 : axis = 1 // train_csv에 있는 'count'열 삭제 
print(x)
y = train_csv['count']                      # train_csv에 있는 'count'열을 y로 설정
print(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.90, random_state=123123,
)
print(x_train.shape, x_test.shape)      #(1021, 9) (438, 9) -> (929, 9) (399, 9)
print(y_train.shape, y_test.shape)      #(1021,) (438,) -> (929,) (399,)


# 나눠 놨으니까 이걸로 트레인


#2. 모델구성
model = Sequential()
model.add(Dense(20, input_dim=9))
model.add(Dense(40))
model.add(Dense(100))
model.add(Dense(20))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=100)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)       # 모델로 예측을 수행하기 위한 함수
y_predict = model.predict(x_test)           # x_test를 넣으면 y_predict 나옴
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)       # (715, 1)

print("===========================================")

######### submission.csv 만들기 (count칼럼에 값만 넣어주면 됨) #############
submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape) # (715, 2) // id, count


submission_csv.to_csv(path + "submission_0105.csv", index=False)        #  to_csv 혹은 to_excel 함수를 사용할 때 'index=False' 추가

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)         
                                      
print("로스 : ", loss)
print("r2 스코어 : " , r2)

'''


# 트레인 결측 제거, 테스트 결측 평균




# 로스 :  2948.28466796875
# r2 스코어 :  0.5548608186449331

# 트레인 사이즈 0.9
# 로스 :  2463.086669921875
# r2 스코어 :  0.5934725909630179

# 로스 :  2470.8466796875
# r2 스코어 :  0.6719312672194586

# 로스 :  2460.9375
# r2 스코어 :  0.6732469455330123

# 로스 :  2640.703857421875
# r2 스코어 :  0.7169629549412184





'''