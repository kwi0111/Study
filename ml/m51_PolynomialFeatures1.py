import numpy as np
from sklearn.preprocessing import PolynomialFeatures # 다항 피쳐

x = np.arange(8).reshape(4, 2)
print(x)
# [[0 1]    -> 0, 0, 1
#  [2 3]    -> 4, 6, 9 
#  [4 5]    -> 16, 20, 25
#  [6 7]]   -> 36, 42, 49
#  컬럼이 증폭되었다. // 원래 컬럼과 수치적인 연관성이 있다. // 

pf = PolynomialFeatures(degree=2, include_bias=False)
x_pf = pf.fit_transform(x)
print(x_pf)
# [[ 0.  1.  0.  0.  1.]    # 제곱, 곱하기, 제곱 ( 컬럼, 데이터 증폭 )
#  [ 2.  3.  4.  6.  9.]
#  [ 4.  5. 16. 20. 25.]
#  [ 6.  7. 36. 42. 49.]]

pf = PolynomialFeatures(degree=3,
                        include_bias=False,   # True 디폴트 : 컬럼 1
                        )
x_pf = pf.fit_transform(x)
print(x_pf)
# [[  0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  2.   3.   4.   6.   9.   8.  12.  18.  27.]
#  [  4.   5.  16.  20.  25.  64.  80. 100. 125.]
#  [  6.   7.  36.  42.  49. 216. 252. 294. 343.]]

print("================== 컬럼 3개 ===========================")
x = np.arange(12).reshape(4, 3)
pf = PolynomialFeatures(degree=2,
                        include_bias=False,   # True 디폴트 : 컬럼 1
                        )
x_pf = pf.fit_transform(x)
print(x_pf)
# [[  0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  9.  10.  11.  81.  90.  99. 100. 110. 121.]]












