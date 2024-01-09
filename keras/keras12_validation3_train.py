import numpy as np
from sklearn.model_selection import train_test_split



#1.데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train = x[0:10]
y_train = y[0:10]
print(x_train)
print(y_train)

x_val = x[10:13]
y_val = y[10:13]
print(x_val)
print(y_val)

x_test = x[13:18]
y_test = y[13:18]
print(x_test)
print(y_test)

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    shuffle=True,
                                                    random_state=123,
                                                    test_size = 0.3
                                                    )
x_test, x_val,y_test, y_val = train_test_split(x_test,
                                               y_test,
                                               shuffle=True,
                                               random_state=123,
                                               test_size=0.3
                                               )
print(x_train, x_test, y_train, y_test)
print(x_test, x_val,y_test, y_val)




# train_test_split로만 잘라라



