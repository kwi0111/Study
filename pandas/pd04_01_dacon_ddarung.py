#https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
from sklearn.svm import LinearSVR


#1. 데이터
path = "c:\\_data\\dacon\\ddarung\\"
# 판다스에서 시리즈(1차원) : 백터 (인덱스가 있다) // 데이터 프레임(2차원) : 행렬

train_csv = pd.read_csv(path + "train.csv", index_col=0) 
test_csv = pd.read_csv(path +"test.csv", index_col=0)
submission_csv = pd.read_csv(path + "submission.csv") 

print(train_csv.shape)      #(1328, 10)
print(test_csv.shape)       # (715, 9)

print(train_csv.info())
print(train_csv.isnull().sum())
print(test_csv.info())
print(test_csv.isnull().sum())

###########################결측치처리########################
train_csv = train_csv.fillna(train_csv.bfill())  # back
print(train_csv.isnull().sum())
test_csv = test_csv.fillna(test_csv.bfill())  # back
print(test_csv.isnull().sum())



################# x와 y를 분리 ###########
x = train_csv.drop(['count',], axis=1)
y = train_csv['count']




def outliers(x):
    quartile_1, q2, quartile_3 = np.percentile(x,[25,50,75])
    
    print('1사분위 :', quartile_1)
    print('q2 :', q2)
    print('3사분위 :', quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr :", iqr)
    lower_bound = quartile_1 - (iqr)   #범위 지정 가능
    upper_bound = quartile_3 + (iqr * 1.5)
    
    return np.where((x>upper_bound) |
                    (x<lower_bound))
    # or -> 두가지 조건 중 하나라도 만족하는게 있으면 리턴
    
    
outliers_loc = outliers(x)
print("이상치의 위치 :", outliers_loc)
print((outliers_loc[0]))    #1513

# import matplotlib.pyplot as plt
# plt.boxplot(outliers_loc)
# plt.show()


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72,  shuffle= False, random_state= 6) #399 #1048 #6


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

#mms = MinMaxScaler()
#mms = StandardScaler()
#mms = MaxAbsScaler()
mms = RobustScaler()

#2. 모델구성
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()  
poly.fit(x_train)  
x_train = poly.transform(x_train)  
x_test = poly.transform(x_test)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
print(x_train.shape)      #(1050, 55)
print(x_test.shape)      #(409, 55)

#3. 컴파일, 훈련
# import datetime
# date= datetime.datetime.now()
# # print(date) #2024-01-17 11:00:58.591406
# # print(type(date)) #<class 'datetime.datetime'>
# date = date.strftime("%m%d-%H%M") #m=month, M=minutes
# # print(date) #0117_1100
# # print(type(date)) #<class 'str'>

# path= 'c:/_data/_save/MCP/_k28/' #경로(스트링data (문자))
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #filename= 에포4자리수-발로스는 소숫점4자리까지 표시. 예)1000-0.3333.hdf5
# filepath = "".join([path, 'k28_4', date, "_", filename]) #""공간에 ([])를 합쳐라.

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

#4. 평가, 예측
loss = model.score(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
# 테스트 데이터에도 동일한 다항식 특성 변환 적용
x_test_submit_poly = poly.transform(test_csv)
# y_submit = model.predict(test_csv)

print("model.score :", loss)
print("R2 스코어 :", r2)
print("걸린 시간 :", round(end_time - start_time, 2), "초")

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


# 테스트 데이터로 예측
y_submit = model.predict(x_test_submit_poly)
y_submit = (y_submit.round(0).astype(int)) #실수를 반올림한 정수로 나타내줌.
'''


####### submission.csv 만들기 (count컬럼에 값만 넣어주면 됨) #####
submission_csv['count'] = y_submit
print(submission_csv)

#submission_csv.to_csv(path + "submission__45.csv", index= False)

path = "c:\\_data\\daicon\\ddarung\\"
import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path + f"submission_{save_time}{rmse:.3f}.csv", index=False)

'''
# model.score : 0.7598985636930471
# R2 스코어 : 0.7598985636930471
# 걸린 시간 : 0.25 초
# RMSE :  38.783530652555996

# 특성 공학
# model.score : 0.7648893344904204
# R2 스코어 : 0.7648893344904204
# 걸린 시간 : 0.92 초
# RMSE :  38.3783349480422