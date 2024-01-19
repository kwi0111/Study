from tensorflow.python.keras.models import Sequential
# from tensorflow.keras.models import Sequential
# from keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D   # cnn 임포트 - 레이어 // 이미지는 Conv2D

model = Sequential()
model.add(Dense(10, input_dim=(3,)))    # 인풋은 (n,3) // 이미지는 4차원 데이터 
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(10,10,1))) # 1개의 흑백 색, kernel_size=(2,2) 얼만큼 자를거냐
# 10 다음 레이어로 전달해주는 아웃풋값
model.add(Dense(5))
model.add(Dense(1))




