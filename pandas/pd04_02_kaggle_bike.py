from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import time
import math
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

#data
path = "C:\\_data\\kaggle\\bike\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submission_csv = pd.read_csv(path+"sampleSubmission.csv")

x = train_csv.drop(['casual','registered','count'],axis=1)
y = train_csv['count']

print(x.shape, y.shape)

def fit_outlier(data):  
    data = pd.DataFrame(data)
    for label in data:
        series = data[label]
        q1 = series.quantile(0.25)      
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + iqr
        lower_bound = q1 - iqr
        
        series[series > upper_bound] = np.nan
        series[series < lower_bound] = np.nan
        print(series.isna().sum())
        series = series.interpolate()
        data[label] = series
        
    data = data.fillna(data.ffill())
    data = data.fillna(data.bfill())
    return data

print(x.isna().sum())
x = fit_outlier(x)
print(x.isna().sum())

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.85,  shuffle= False, random_state= 123)


#2. 모델구성
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
scaler = MinMaxScaler()
scaler.fit(x_train)  
x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures()  
poly.fit(x_train)  
x_train = poly.transform(x_train)  
x_test = poly.transform(x_test)

#model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

#fit
model.fit(x_train,y_train)

#evaluate & predict 
loss = model.score(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
y_submit = poly.transform(test_csv)
# y_submit = model.predict(test_csv)

#### CSV파일 생성 ####
submission_csv['count'] = y_submit
dt = datetime.datetime.now()
# submission_csv.to_csv(path+f"submission_{dt.day}day{dt.hour}-{dt.minute}.csv",index=False)
# submission_csv.to_csv(path+f"submission_{dt.hour}-{dt.minute}_loss{loss}.csv",index=False)


#### 음수 개수와 RMSLE출력 ####
num_of_minus = submission_csv[submission_csv['count']<0].count()
# print(num_of_minus['count'])

def RMSLE(y_test,y_predict):
    return np.sqrt(mean_squared_log_error(y_test,y_predict))

if num_of_minus['count'] == 0:    
    print("RMSLE: ",RMSLE(y_test,y_predict))
else:
    print("음수갯수: ",num_of_minus['count'])
    for i in range(len(y_submit)):
        if y_submit[i] < 0:
            y_submit[i] = 0
    
# loss=0.34769600159036584
# r2=0.34769600159036584
# RMSLE:  1.2953199766566694

# RMSLE:  1.2238634030450557
