from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.7,
    
    fill_mode='nearest',
    
)

augumet_size = 100      # 가상 이미지의 수 (변수)
# print(x_train[0].shape) # (28, 28)
# plt.imshow(x_train[0])
# plt.show()

x_data = train_datagen.flow(    # flow에 원래 x,y 들어감
    np.tile(x_train[0].reshape(28*28),augumet_size).reshape(-1,28,28,1), # x // (100, 28, 28, 1)
    np.zeros(augumet_size),                                              # y // 구색 맞춤이다.
    batch_size=100,
    shuffle=True
    ) # .next()    # 0번째 껏만 출력 x데이터에

# print(x_data)
# print(x_data.shape) # 'tuple' object has no attribute 'shape' 
# // 튜플형태라서 에러 왜냐하면 flow에서 튜플형태로 반환했음. // 넘파이로 확인해야함 // len으로 확인해야함 
print(x_data[0][0].shape)  # (100, 28, 28, 1)   next() 안쓰면 1개씩 밀린다.
print(x_data[0][1].shape)  # (100, )
print(np.unique(x_data[0][1], return_counts=True)) # (array([0.]), array([100], dtype=int64))

print(x_data[0][0][1].shape)   # (28, 28, 1)

plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7, i+1)   # 0번째부터 시작
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap='gray')
plt.show()

# 데이터 수집하지 못할때 데이터 증폭 ,,








