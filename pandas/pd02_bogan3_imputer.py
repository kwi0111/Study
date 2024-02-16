import numpy as np
import pandas as pd


data = pd.DataFrame([
    [2, np.nan ,6 ,8, 10],
    [2, 4 ,np.nan ,8, np.nan],
    [2, 4 ,6 ,8, 10],
    [np.nan, 4 ,np.nan ,8, np.nan],
])

data = data.T
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

from sklearn.impute import SimpleImputer, KNNImputer    # 전략에 따라 누락된 값을 대체 // 가장 단순한 방법은 평균, 중앙값, 최빈값 등의 대표값을 사용
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = SimpleImputer()
data2 = imputer.fit_transform(data) # 디폴트 평균값.
print(data2)

imputer = SimpleImputer(strategy='mean')
data3 = imputer.fit_transform(data) # 평균
print(data3)

imputer = SimpleImputer(strategy='median')
data4 = imputer.fit_transform(data) # 중위
print(data4)

imputer = SimpleImputer(strategy='most_frequent')
data5 = imputer.fit_transform(data) # 가장 자주 나오는놈 // 컬럼 별로 넣어야함.
print(data5)

imputer = SimpleImputer(strategy='constant')
data6 = imputer.fit_transform(data) # 상수 0 들어간다. 
print(data6)

imputer = SimpleImputer(strategy='constant', fill_value=777)
data7 = imputer.fit_transform(data) # 특정수
print(data7)


print("========== KNNImputer ==============")   # 범위안 뭐가 더 많은지 판단 
imputer = KNNImputer()  # KNN알고리즘으로 결측치 처리
data8 = imputer.fit_transform(data)
print(data8)

imputer = IterativeImputer()    # 선형 회귀 알고리즘
data9 = imputer.fit_transform(data)
print(data9)

print(np.__version__)   # 1.26.3 에서 mice 오류   
print(np.__version__)   # 1.22.4 에서 mice 정상 작동

from impyute.imputation.cs import mice  # 다중 선형 방식 // predict 방식
aaa = mice(data.values,
           n=10,
           seed=777)
print(aaa)



