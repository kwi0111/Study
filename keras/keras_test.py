import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv1D, Flatten, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical #

import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터
path = "C:\\_data\\dacon\\dechul\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0 )
print(train_csv.shape)  # (96294, 14)
test_csv = pd.read_csv(path + "test.csv", index_col=0 )
print(test_csv.shape)  # (64197, 13)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv.shape)  # (64197, 2)

# print(train_csv.info())
# a = train_csv['대출금액'] / train_csv['총상환원금']
# print(a[23])


# print(train_csv.shape, test_csv.shape) #(96294, 14) (64197, 13)
# print(train_csv.columns, test_csv.columns,sep='\n',end="\n======================\n")
# Index(['대출금액', '대출기간', '근로기간', '주택소유상태', '연간소득', '부채_대비_소득_비율', '총계좌수', '대출목적',
#        '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수', '대출등급'],
#       dtype='object')
# Index(['대출금액', '대출기간', '근로기간', '주택소유상태', '연간소득', '부채_대비_소득_비율', '총계좌수', '대출목적',
#        '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수'],
#       dtype='object')
# print(np.unique(train_csv['대출등급'],return_counts=True))
# print(np.unique(train_csv['주택소유상태'],return_counts=True))


# print(np.unique(test_csv['주택소유상태'],return_counts=True),end="\n======================\n")
# (array(['ANY', 'MORTGAGE', 'OWN', 'RENT'], dtype=object), array([    1, 47934, 10654, 37705], dtype=int64))
# (array(['MORTGAGE', 'OWN', 'RENT'], dtype=object), array([31739,  7177, 25281], dtype=int64))

# print(np.unique(train_csv['대출목적'],return_counts=True))
# print(np.unique(test_csv['대출목적'],return_counts=True),end="\n======================\n")
# (array(['기타', '부채 통합', '소규모 사업', '신용 카드', '의료', '이사', '자동차', '재생 에너지',
#        '주요 구매', '주택', '주택 개선', '휴가'], dtype=object), array([ 4725, 55150,   787, 24500,  1039,   506,   797,    60,  1803,
#          301,  6160,   466], dtype=int64))
# (array(['결혼', '기타', '부채 통합', '소규모 사업', '신용 카드', '의료', '이사', '자동차',
#        '재생 에너지', '주요 구매', '주택', '주택 개선', '휴가'], dtype=object), array([    1,  3032, 37054,   541, 16204,   696,   362,   536,    29,
#         1244,   185,  4019,   294], dtype=int64))

# print(np.unique(train_csv['대출등급'],return_counts=True),end="\n======================\n")
# (array(['A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype=object), array([16772, 28817, 27623, 13354,  7354,  1954,   420], dtype=int64))

# train_csv = train_csv[train_csv['주택소유상태'] != 'ANY'] #ANY은딱 한개 존재하기에 그냥 제거
# test_csv = test_csv[test_csv['대출목적'] != '결혼']
# test_csv.loc[test_csv['대출목적'] == '결혼' ,'대출목적'] = '기타' #결혼은 제거하면 개수가 안맞기에 기타로 대체

# x.loc[x['type'] == 'red', 'type'] = 1
# print(np.unique(train_csv['주택소유상태'],return_counts=True))
# print(np.unique(test_csv['주택소유상태'],return_counts=True),end="\n======================\n")
# print(np.unique(train_csv['대출목적'],return_counts=True))
# print(np.unique(test_csv['대출목적'],return_counts=True),end="\n======================\n")




# 라벨 엔코더
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder() # 대출기간, 대출목적, 근로기간, 주택소유상태 // 라벨 인코더 : 카테고리형 피처를 숫자형으로 변환
train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = le.fit_transform(train_csv['대출목적'])
train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
# train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])

test_csv['주택소유상태'] = le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = le.fit_transform(test_csv['대출목적'])
test_csv['대출기간'] = test_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
# test_csv['근로기간'] = le.fit_transform(test_csv['근로기간'])

train_csv['대출등급'] = le.fit_transform(train_csv['대출등급']) # 마지막에 와야함



for i in range(len(train_csv['근로기간'])):
    data = train_csv['근로기간'].iloc[i]
    if data == 'Unknown':
        train_csv['근로기간'].iloc[i] = np.NaN
    elif data == '10+ years' or data == '10+years':
        train_csv['근로기간'].iloc[i] = int(30)
    elif data == '< 1 year' or data == '<1 year':
        train_csv['근로기간'].iloc[i] = int(0)
    else:
        train_csv['근로기간'].iloc[i] = int(data.split()[0])
    
train_csv['근로기간'] = train_csv['근로기간'].fillna(train_csv['근로기간'].mean())

for i in range(len(test_csv['근로기간'])):
    data = test_csv['근로기간'].iloc[i]
    if data == 'Unknown':
        test_csv['근로기간'].iloc[i] = np.NaN
    elif data == '10+ years' or data == '10+years':
        test_csv['근로기간'].iloc[i] = int(30)
    elif data == '< 1 year' or data == '<1 year':
        test_csv['근로기간'].iloc[i] = int(0)
    else:
        test_csv['근로기간'].iloc[i] = int(data.split()[0])
    
test_csv['근로기간'] = test_csv['근로기간'].fillna(test_csv['근로기간'].mean())




#신규고객 제거
# train_csv = train_csv[train_csv['총상환이자'] != 0.0 ]





print(train_csv.describe)
print(test_csv.describe)

print(train_csv.shape)
print(test_csv.shape)
print(train_csv.dtypes)
print(test_csv.dtypes)

# x와 y를 분리
x = train_csv.drop(['대출등급','총계좌수','대출목적',], axis=1)
y = train_csv['대출등급']
test_csv = test_csv.drop(['총계좌수','대출목적',], axis=1)
print(x.shape, y.shape) # (96294, 13) (96294,)
print(pd.value_counts(y))
# 대출등급
# 1    28817
# 2    27623
# 0    16772
# 3    13354
# 4     7354
# 5     1954
# 6      420

# mms = MinMaxScaler()
# mms.fit(x)
# x = mms.transform(x)
# test_csv=mms.transform(test_csv)

y = np.reshape(y, (-1,1)) 
# y = np.array()


ohe = OneHotEncoder(sparse = False)
ohe.fit(y)
y_ohe = ohe.transform(y)
print(y.shape)  



# y_ohe = pd.get_dummies(y, dtype='int')
# print(y_ohe)   
# print(x.shape, y.shape)   # (96294, 13) (96294, 1) // (96294, ) 벡터 형태 -> reshape를 이용해 행렬로 바꿔줘야함


x_train, x_test, y_train, y_test = train_test_split(
                                                    x,
                                                    y_ohe,             
                                                    train_size=0.87,
                                                    random_state=2024,
                                                    stratify=y_ohe,
                                                    shuffle=True,
                                                    )
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# scaler = MinMaxScaler() # 클래스 정의
scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# x_train = x_train.reshape(-1, 11, 1)
# x_test = x_test.reshape(-1,11, 1)
# test_csv = test_csv.reshape(-1, 11, 1)


# 2. 모델 구성 
# model = Sequential()
# model.add(Conv1D(64, 2, padding='same', input_shape = (11,1), activation='relu'))
# model.add(Conv1D(32, 2, padding='same', activation='relu'))
# model.add(Conv1D(16, 2, padding='same', activation='relu'))
# model.add(Flatten())
# model.add(Dense(520, activation='relu'))
# model.add(Dropout(0.05))
# model.add(Dense(230, activation='relu'))
# model.add(Dense(130, activation='relu'))
# model.add(Dense(60, activation='relu'))
# model.add(Dropout(0.05))
# model.add(Dense(7, activation='softmax'))  

model = Sequential()
model.add(Dense(12, input_dim=11, activation='swish'))
model.add(Dense(80, activation='swish')) # 80
model.add(Dropout(0.02))
model.add(Dense(70, activation='swish'))
model.add(Dropout(0.02))
model.add(Dense(10, activation='swish'))
model.add(Dense(7, activation='swish'))
model.add(Dense(5, activation='swish'))
model.add(Dense(100, activation='swish'))
model.add(Dense(5, activation='swish'))
model.add(Dense(5, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(100, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(4, activation='swish'))
model.add(Dense(8, activation='swish'))
model.add(Dense(7, activation='softmax'))






#2. 모델구성(함수형)
# input1 = Input(shape=(13, ))
# dense1 = Dense(10, activation='swish')(input1)
# dense2 = Dense(80, activation='swish')(dense1)
# dense3 = Dense(60, activation='swish')(dense2)
# dense4 = Dense(20, activation='swish')(dense3)
# dense5 = Dense(10, activation='swish')(dense4)
# dense6 = Dense(5, activation='swish')(dense5)
# dense7 = Dense(10, activation='swish')(dense6)
# dense8 = Dense(10, activation='swish')(dense7)
# dense9 = Dense(10, activation='swish')(dense8)
# dense10 = Dense(5, activation='swish')(dense9)
# dense11 = Dense(10, activation='swish')(dense10)
# output1 = Dense(7, activation='softmax')(dense11)
# model = Model(inputs = input1, outputs = output1)


#3.컴파일, 훈련
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
MCP_path = "../_data/_save/MCP/"
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([MCP_path, 'k23_', date, '_', filename])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=300,
                verbose=2,
                restore_best_weights=True
                )
# mcp = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       verbose=1,
#                       save_best_only=True,
#                       filepath=filepath,
#                       )

model.fit(x_train, y_train, epochs=15000, batch_size = 1324,
                validation_split=0.16,  
                callbacks=[es],
                verbose=2
                )

#4.평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
arg_pre = np.argmax(y_predict, axis=1)    #  argmax : NumPy 배열에서 가장 높은 값을 가진 값의 인덱스를 반환
arg_test = np.argmax(y_test, axis=1)
y_submit = model.predict(test_csv)
submit = np.argmax(y_submit, axis=1)
submitssion = le.inverse_transform(submit)
      
submission_csv['대출등급'] = submitssion
y_predict = ohe.inverse_transform(y_predict)
y_test = ohe.inverse_transform(y_test)
f1 = f1_score(y_test, y_predict, average='macro')
acc = accuracy_score(y_test, y_predict)
print("로스 : ", results[0])  
print("acc : ", results[1])  
print("f1 : ", f1)  
submission_csv.to_csv(path + "submission_0202_1.csv", index=False)



'''
로스 :  0.16252931952476501
acc :  0.9449636340141296
f1 :  0.9268314268128603


#3 트 0.84 발 0.17 레이어 추가 
로스 :  0.17536622285842896
acc :  0.9391225576400757
f1 :  0.9155153131294058

#4 트 0.92 발 0.18




# 모델 바꿈 1324
로스 :  0.16004136204719543
acc :  0.9447040557861328
f1 :  0.9358053156502212

모델 레이어 100 100
로스 :  0.1952265202999115
acc :  0.9373052716255188
f1 :  0.9256020112240373

로스 :  0.20408895611763
acc :  0.9328920245170593
f1 :  0.924158660092006

로스 :  0.20736417174339294
acc :  0.9321131706237793
f1 :  0.9129629504600116

로스 :  0.17329061031341553
acc :  0.943665623664856
f1 :  0.9389693180276977

로스 :  0.1596868932247162
acc :  0.945353090763092
f1 :  0.9313242487216843

'''












