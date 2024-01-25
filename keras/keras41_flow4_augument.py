from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.


train_datagen = ImageDataGenerator(
    # rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.7,
    
    fill_mode='nearest',
    
)

augumet_size = 40000      # 가상 이미지의 수 (변수)

randidx = np.random.randint(x_train.shape[0], size=augumet_size)    # 60000중에서 40000개의 숫자를 뽑아내라.
                                                                    # np.random.randint(60000, 40000)

print(randidx)  # [15687 24705 24984 ... 28019 44512 18162]
print(np.min(randidx), np.max(randidx)) # 2 59998

x_augumented = x_train[randidx].copy()  # 메모리 원데이터에 영향을 미치지 않기위해서 사용
y_augumented = y_train[randidx].copy()   # 키벨류로 쌍이 맞음

# print(x_augumented)
# print(x_augumented.shape)   # (40000, 28, 28)
# print(y_augumented)
# print(y_augumented.shape)   # (40000,)

# x_augumented = x_augumented.reshape(-1, 28, 28, 1)

x_augumented = x_augumented.reshape(
    x_augumented.shape[0], x_augumented.shape[1], x_augumented.shape[2], 1)

x_augumented = train_datagen.flow(
    x_augumented, y_augumented,
    batch_size=augumet_size,
    shuffle=False,
    
).next()[0]    # 4만개 변환

print(x_augumented)
print(x_augumented.shape)   # (40000, 28, 28, 1)


print(x_train.shape)    #(60000, 28, 28)
x_train = x_train.reshape(60000, 28, 28, 1)   
x_test = x_test.reshape(10000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augumented))        # 사슬 같이 잇다
y_train = np.concatenate((y_train, y_augumented))        # 사슬 같이 잇다
print(x_train.shape, y_train.shape)                      # (100000, 28, 28, 1) (100000,)

# 원래와 비교 엠미스트 패9션













# # print(x_train[0].shape) # (28, 28)
# # plt.imshow(x_train[0])
# # plt.show()

# x_data = train_datagen.flow(    # flow에 원래 x,y 들어감
#     np.tile(x_train[0].reshape(28*28),augumet_size).reshape(-1,28,28,1), # x // (100, 28, 28, 1)
#     np.zeros(augumet_size),                                              # y // 구색 맞춤이다.
#     batch_size= augumet_size,
#     shuffle=True
#     ).next()    # 0번째 껏만 출력 x데이터에

# # print(x_data)
# # print(x_data.shape) # 'tuple' object has no attribute 'shape' 
# # // 튜플형태라서 에러 왜냐하면 flow에서 튜플형태로 반환했음. // 넘파이로 확인해야함 // len으로 확인해야함 
# print(x_data[0].shape)  # (100, 28, 28, 1)
# print(x_data[1].shape)  # (100, )
# print(np.unique(x_data[1], return_counts=True)) # (array([0.]), array([100], dtype=int64))

# # print(x_data[0][0].shape)   # (28, 28, 1)

# plt.figure(figsize=(7,7))
# for i in range(49):
#     plt.subplot(7,7, i+1)   # 0번째부터 시작
#     plt.axis('off')
#     plt.imshow(x_data[0][i], cmap='gray')
# plt.show()







