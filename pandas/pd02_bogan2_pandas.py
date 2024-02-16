import numpy as np
import pandas as pd

#### 결측치 해결 ####
data = pd.DataFrame([
    [2, np.nan ,6 ,8, 10],
    [2, 4 ,np.nan ,8, np.nan],
    [2, 4 ,6 ,8, 10],
    [np.nan, 4 ,np.nan ,8, np.nan],
])

data = data.T
data.columns = ['x1', 'x2', 'x3', 'x4']
# print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# 결측치 확인
print(data.isnull())    # True가 결측치
#       x1     x2     x3     x4
# 0  False  False  False   True
# 1   True  False  False  False
# 2  False   True  False   True
# 3  False  False  False  False
# 4  False   True  False   True
print(data.isnull().sum())
# x1    1
# x2    2
# x3    0
# x4    3
print(data.info())
#    Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   x1      4 non-null      float64
#  1   x2      3 non-null      float64
#  2   x3      5 non-null      float64
#  3   x4      2 non-null      float64

#1.결측치 삭제
print(data.dropna())   # 행값 빼버렷다 
#     x1   x2   x3   x4
# 3  8.0  8.0  8.0  8.0
print(data.dropna(axis=1))  # axis = 0 디폴트 // 열값 빼버렸다.

# #2-1. 특정값 - 평균
means = data.mean()
print(means)
data2 = data.fillna(means)
print(data2)

# #2-2 특정값 - 중위
med = data.median()
print(med)
data3 = data.fillna(med)
print(data3)

# #2-3 특정값 - 0 채우기 / 임의의값 채우기
data4 = data.fillna(0)
print(data4)
data4_2 = data.fillna(777)
print(data4_2)
#2-4 특정값 - ffill
data5 = data.fillna(method='ffill')
data5 = data.ffill()
print(data5)

#2-4 특정값 - bfill
data6 = data.fillna(method='bfill')
# data6 = data.bfill()
print(data6)
############### 특정 칼럼만 #################
means = data['x1'].mean()
print(means)    # 6.5
meds = data['x4'].median()
print(meds)     # 6.0


data['x1'] = data['x1'].fillna(means)   # x1 칼럼은 평균값
data['x4'] = data['x4'].fillna(meds) # x4 칼럼은 중위값
data['x2'] = data['x2'].ffill() # x2 칼럼은 앞값

print(data)    

'''
'''



