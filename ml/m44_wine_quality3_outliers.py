import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier


#1. 데이터
path = "C:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


x = train_csv.drop(['quality'], axis= 1)
y = train_csv['quality']

def outliers(x):
    x_numeric = x.drop(columns=['type'])  # 'type' 열 제외하고 숫자형 데이터만 사용
    quartile_1, q2, quartile_3 = np.percentile(x_numeric,[25,50,75])
    
    print('1사분위 :', quartile_1)
    print('q2 :', q2)
    print('3사분위 :', quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr :", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)   #범위 지정 가능
    upper_bound = quartile_3 + (iqr * 1.5)
    
    # 'type' 열의 문자열 값을 숫자로 변환하여 처리
    x['type'] = LabelEncoder().fit_transform(x['type'])

    return np.where((x_numeric>upper_bound) |
                    (x_numeric<lower_bound))
    # or -> 두가지 조건 중 하나라도 만족하는게 있으면 리턴

outliers_loc = outliers(x)
print("이상치의 위치 :", outliers_loc)
# print("이상치 개수 :", num_outliers)
print((outliers_loc[0]))    #

# import matplotlib.pyplot as plt
# plt.boxplot(x)
# plt.show()

y = LabelEncoder().fit_transform(y)

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0

test_csv.loc[test_csv['type'] == 'red', 'type'] = 1 
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

x_train, x_test, y_train, y_test = train_test_split(
    x, y , random_state=1, train_size=0.8,
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = RandomForestClassifier()

#.3 훈련
model.fit(x_train, y_train,
          )

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score : ', acc)


'''
최종점수 :  0.6790909090909091
acc_score :  0.6790909090909091
'''




