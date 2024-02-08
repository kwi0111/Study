import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping

#1. 데이터
path = "C:\\_data\\sihum\\"
ss = pd.read_csv(path + '삼성 240205.csv', index_col=0, encoding='EUC-KR')
am = pd.read_csv(path + '아모레 240205.csv', index_col=0, encoding='EUC-KR')

ss = ss[:540]
am = am[:540]
# print(ss)
# print(am)
# print(ss.shape) # (10296, 16)
# print(am.shape) # (4350, 16)

# print(ss.info())
# print(ss.isna().sum()) 
# print(am.isna().sum()) 
# 삼성전자 데이터
ss = ss.drop(['신용비', '외인비'], axis=1)
# print(ss)
ss['전일비'] = ss['전일비'].replace({"▼":0, "▲":1, " ":0, '↑':1, '↓':0}).astype(float)
ss['등락률'] = ss['등락률'].astype(float) 
# print(ss)
# print(ss['전일비'])

# 아모레 데이터
am = am.drop(['신용비', '외인비'], axis=1)
am['전일비'] = am['전일비'].replace({"▼":0, "▲":1, " ":0, '↑':1, '↓':0}).astype(float)
am['등락률'] = am['등락률'].astype(float) 
# print(ss.shape)  # (270, 14)
# print(ss.info())

# print(ss.isna().sum()) 
# # ss = ss.dropna()
# print(am.isna().sum()) 
        
for i in range(len(ss.index)):
    for j in range(len(ss.iloc[i])):
        # 현재 셀의 값이 문자열이고 비어 있지 않은 경우에만 변환을 수행합니다.
        if isinstance(ss.iloc[i, j], str) and ss.iloc[i, j].strip():
            # 문자열에 '↑'가 포함되어 있는 경우, 해당 값을 0으로 대체합니다.
            if '↑' in ss.iloc[i, j]:
                ss.iloc[i, j] = 1
            # 문자열에 '↓'가 포함되어 있는 경우, 해당 값을 1로 대체합니다.
            elif '↓' in ss.iloc[i, j]:
                ss.iloc[i, j] = 0
            else:
                # 쉼표(,)를 제거하고 정수형으로 변환합니다.
                ss.iloc[i, j] = float(ss.iloc[i, j].replace(',', ''))
                
for i in range(len(am.index)):
    for j in range(len(am.iloc[i])):
        # 현재 셀의 값이 문자열이고 비어 있지 않은 경우에만 변환을 수행합니다.
        if isinstance(am.iloc[i, j], str) and am.iloc[i, j].strip():
            # 문자열에 '↑'가 포함되어 있는 경우, 해당 값을 0으로 대체합니다.
            if '↑' in am.iloc[i, j]:
                am.iloc[i, j] = 0
            # 문자열에 '↓'가 포함되어 있는 경우, 해당 값을 1로 대체합니다.
            elif '↓' in am.iloc[i, j]:
                am.iloc[i, j] = 1
            else:
                # 쉼표(,)를 제거하고 정수형으로 변환합니다.
                am.iloc[i, j] = float(am.iloc[i, j].replace(',', ''))
                
ss = ss[:540].astype(np.float32)
am = am[:540].astype(np.float32)
# print(ss.info())
# print(ss.info())
# print(am.info())


# print(ss.shape, am.shape)  # (270, 14) (270, 14)
# print(type(am))   # <class 'pandas.core.frame.DataFrame'>
# # ss = ss.values
# # am = am.values
# # print(type(am))   # <class 'numpy.ndarray'>
# # print(ss)

# 데이터 전처리
ss_cols = ss.columns
am_cols = am.columns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler1 = MinMaxScaler()
ss = scaler1.fit_transform(ss)

scaler2 = MinMaxScaler()
am = scaler2.fit_transform(am)
print("="*100)
print(ss)

ss = pd.DataFrame(ss,columns=ss_cols)
am = pd.DataFrame(am,columns=am_cols)
print(ss.shape)

size = 30
def split_xy(dataset, size, y_col):     
    result_x = []
    result_y = []
    
    for i in range(len(dataset) - (size + 1)):
        result_x.append(dataset[i:i+size])
        y_row = dataset.iloc[i+size+1]
        result_y.append(y_row[y_col])
    return np.array(result_x), np.array(result_y)

x1, y1 = split_xy(ss, size, '시가')
x2, y2 = split_xy(am, size, '종가')

# print(ss)
# print(x.shape, y.shape) # 



# 데이터셋 나누기
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, x2_train, x2_test, y2_train, y2_test = train_test_split(x1, y1, x2, y2, train_size=0.86,
                                                                                              shuffle=False,
                                                                                              random_state=2024)

print(x1.shape, y1.shape) # (265, 5, 14) (265,)
print(x2.shape, y2.shape) # (265, 5, 14) (265,)

from keras.models import Model
from keras.layers import Dense, Input, concatenate, LSTM

#2-1. 삼성 모델
input1 = Input(shape=(30, 14))
dense1 = Dense(100, activation='relu')(input1) 
dense2 = Dense(100, activation='relu')(dense1)
dense3 = Dense(100, activation='relu')(dense2)
output1 = Dense(100, activation='relu')(dense3)

#2-2. 아모레 모델
input11 = Input(shape=(30, 14))
dense11 = Dense(100, activation='relu')(input11)
dense12 = Dense(100, activation='relu')(dense11)
dense13 = Dense(100, activation='relu')(dense12)
output11 = Dense(100, activation='relu')(dense13)


#2-3. concatnate 사슬처럼 엮다.
merge1 = concatenate([output1, output11])
lstm = LSTM(20)(merge1)
merge2 = Dense(14)(lstm)
merge3 = Dense(14)(merge2)
last_output1 = Dense(1)(merge3) 
last_output2 = Dense(1)(merge3) 

model = Model(inputs = [input1, input11], outputs = [last_output1, last_output2])

model.summary()
# print(y1_train[0])
# print(y2_train[0])

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=100,
                verbose=2,
                restore_best_weights=True
                )
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=50, batch_size=6, verbose=2)

#4. 평가 및 예측
results = model.evaluate([x1_test, x2_test], [y1_test, y2_test])
y_predict = model.predict([x1_test,x2_test])
# y_predict1=model.predict(x1_test)
# y_predict2=model.predict(x2_test)
y_predict = model.predict([x1_test[0].reshape(1,30,14), x2_test[0].reshape(1,30,14)])

print(x1_test[0])
print(y_predict)

y1_temp = np.zeros([1,14])
y1_temp[0][0] = y_predict[0]

y_predict2 = scaler1.inverse_transform(y1_temp)
print("삼성",y_predict2[0][0])

y2_temp = np.zeros([1,14])
y2_temp[0][0] = y_predict[1]

y_predict22 = scaler2.inverse_transform(y2_temp)
print("아모레",y_predict22[0][0])

''' 
print("loss : ", results)
print("삼성전자 시가 :" , y1_test[0] , '/ 예측가 : ', y_predict2[0][0])
print("아모레 종가 :" , y2_test[0] , '/ 예측가 : ', y_predict2[1][0])

print(y1_test.shape)
'''

# for i in range(14):
    #  print("종가 :" , y1_test[i] , '/ 예측가 : ', y_predict[i]) 

# print(y_predict)
# print(len(x1_test))
# print(y)
# print(y_train)

'''
'''
'''


'''
