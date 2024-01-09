import numpy as np



#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])

x_train = x[0:9]
y_train = y[0:9]
print(x_train)
print(y_train)

x_val = x[10:13]
y_val = y[10:13]
print(x_val)
print(y_val)

x_test = x[14:17]
y_test = y[14:17]
print(x_test)
print(y_test)


# 1~10 트레인 11 12 13  발리데이션 14 15 16 17 테스트