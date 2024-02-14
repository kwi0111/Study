#https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import HalvingGridSearchCV


#1. 데이터
path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0) # \ \\ / // 다 가능( 예약어 사용할때 두개씩 사용) 인덱스컬럼은 0번째 컬럼이다라는뜻.
test_csv = pd.read_csv(path +"test.csv", index_col=0)
submission_csv = pd.read_csv(path + "submission.csv") 

print(train_csv.info())
train_csv = train_csv.fillna(train_csv.mean())  #결측치가 하나라도 있으면 행전체 삭제됨.
test_csv = test_csv.fillna(test_csv.mean())   # (0,mean)

test_csv = test_csv.drop(['hour_bef_humidity','hour_bef_windspeed'], axis=1)   # (0,mean)

print(train_csv.shape)      #(1328, 10)

################# x와 y를 분리 ###########
x = train_csv.drop(['count',], axis=1)
y = train_csv['count']

print(x.info())

''' 25퍼 미만 열 삭제 '''
# columns = datasets.feature_names
columns = x.columns
x = pd.DataFrame(x,columns=columns)
print("x.shape",x.shape)
# ''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
fi_str = "0.34626618 0.09873444 0.36363754 0.01909325 0.02808662 0.04098203\
 0.04632215 0.03015554 0.02672229"
 
''' str에서 숫자로 변환하는 구간 '''
fi_str = fi_str.split()
fi_float = [float(s) for s in fi_str]
print(fi_float)
fi_list = pd.Series(fi_float)

''' 25퍼 미만 인덱스 구하기 '''
low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
print('low_idx_list',low_idx_list)

''' 25퍼 미만 제거하기 '''
low_col_list = [x.columns[index] for index in low_idx_list]
# 이건 혹여 중복되는 값들이 많아 25퍼이상으로 넘어갈시 25퍼로 자르기
if len(low_col_list) > len(x.columns) * 0.25:   
    low_col_list = low_col_list[:int(len(x.columns)*0.25)]
print('low_col_list',low_col_list)
x.drop(low_col_list,axis=1,inplace=True)
print("after x.shape",x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8)

print(x_train.info())
# train 9개중에서 7개로 만들었으니까 test도 7개로 만들어 줘야함
#  0   hour                    1167 non-null   int64
#  1   hour_bef_temperature    1167 non-null   float64
#  2   hour_bef_precipitation  1167 non-null   float64
#  3   hour_bef_visibility     1167 non-null   float64
#  4   hour_bef_ozone          1167 non-null   float64
#  5   hour_bef_pm10           1167 non-null   float64
#  6   hour_bef_pm2.5          1167 non-null   float64

#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1459 non-null   float64
#  2   hour_bef_precipitation  1459 non-null   float64
#  3   hour_bef_windspeed      1459 non-null   float64
#  4   hour_bef_humidity       1459 non-null   float64
#  5   hour_bef_visibility     1459 non-null   float64
#  6   hour_bef_ozone          1459 non-null   float64
#  7   hour_bef_pm10           1459 non-null   float64
#  8   hour_bef_pm2.5          1459 non-null   float64

# scaler = MinMaxScaler()
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from xgboost import XGBRegressor
model = XGBRegressor()

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

from sklearn.metrics import r2_score
print('r2_score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
y_submit = model.predict(test_csv)

print("걸린시간 :", round(end_time - start_time, 2), "초")
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
y_submit = (y_submit.round(0).astype(int)) #실수를 반올림한 정수로 나타내줌.
print (type(model).__name__, model.feature_importances_)

# ####### submission.csv 만들기 (count컬럼에 값만 넣어주면 됨) #####
# submission_csv['count'] = y_submit
# print(submission_csv)


# path = "c:\\_data\\dacon\\ddarung\\"
# import time as tm
# ltm = tm.localtime(tm.time())
# save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
# submission_csv.to_csv(path + f"submission_{save_time}{rmse:.3f}.csv", index=False)


