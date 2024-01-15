import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
path = "C:\\_data\\dacon\\dechul\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0 )
print(train_csv.shape)  # (96294, 14)
test_csv = pd.read_csv(path + "test.csv", index_col=0 )
print(test_csv.shape)  # (64197, 13)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv.shape)  # (64197, 2)

# train_csv[train_csv.iloc[:,:] == 'Unknown'] = np.NaN
# train_csv.isnull().sum() # 변수별 결측치의 갯수
# print(train_csv.isnull().sum())
# train_csv['근로기간'] = train_csv['근로기간'].fillna(train_csv['근로기간'].mean())     # train에서 없어진 결측치를 평균인 중간값으로 채움.
# print(train_csv.info())


# 라벨 엔코더
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # 대출기간, 대출목적, 근로기간, 주택소유상태 // 라벨 인코더 : 카테고리형 피처를 숫자형으로 변환
train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])
train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = le.fit_transform(train_csv['대출목적'])
train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int)

test_csv['근로기간'] = le.fit_transform(test_csv['근로기간'])
test_csv['주택소유상태'] = le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = le.fit_transform(test_csv['대출목적'])
test_csv['대출기간'] = test_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
train_csv['대출등급'] = le.fit_transform(train_csv['대출등급'])

print(train_csv.describe)
print(test_csv.describe)

print(train_csv.shape)
print(test_csv.shape)
print(train_csv.dtypes)
print(test_csv.dtypes)

# x와 y를 분리
x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']
print(x.shape)
print(y.shape)
print(pd.value_counts(y))

mms = MinMaxScaler()
mms.fit(x)
x = mms.transform(x)
test_csv=mms.transform(x)

# 대출등급
# 1    28817
# 2    27623
# 0    16772
# 3    13354
# 4     7354
# 5     1954
# 6      420

y_ohe = pd.get_dummies(y, dtype='int')
print(y_ohe)    # [96294 rows x 7 columns]
print(x.shape, y.shape)   # (96294, 13) (96294,)


x_train, x_test, y_train, y_test = train_test_split(
                                                    x,
                                                    y_ohe,             
                                                    train_size=0.8,
                                                    random_state=123,
                                                    stratify=y_ohe,
                                                    shuffle=True,
                                                    )

#2. 모델 구성 
model = Sequential()
model.add(Dense(200, input_dim=13))
model.add(Dense(150))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(7, activation='softmax'))


#3.컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=500,
                verbose=1,
                restore_best_weights=True
                )

model.fit(x_train, y_train, epochs=10, batch_size=1000,
                validation_split=0.2,
                callbacks=[es]
                )

#4.평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_arg_pre = np.argmax(model.predict(x_test))    #  argmax : NumPy 배열에서 가장 높은 값을 가진 값의 인덱스를 반환
y_arg_test = np.argmax(y_test, axis=1)
submit = np.argmax(model.predict(test_csv), axis=1)
submitssion = le.inverse_transform(submit)
      
submission_csv['대출등급'] = submitssion
# f1 = f1_score(y_arg_test, y_arg_pre, average=None)
# print("f1 스코어 : ", f1)
print("로스 : ", results[0])  
print("acc : ", results[1])  

# y_test = np.argmax(y_test, axis=1)
# print(y_test)
# print(y_predict)
# print(y_predict.shape, y_test.shape)  


submission_csv.to_csv(path + "submission_0115.csv", index=False)
'''
'''



