import sys  #
import tensorflow as tf
print('텐서플로우 버젼 : ', tf.__version__) # 텐서플로우 버젼 :  2.9.0
print('파이썬 버젼 : ', sys.version)        # 파이썬 버젼 :  3.9.18 
import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array     # 이미지 가져온다.
from tensorflow.keras.preprocessing.image import load_img     # 이미지 수치화

path = "c:\_data\cat_and_dog\Train\Cat\\1.jpg"

img = load_img(path,
                target_size=(150,150),
                     )

print(img) # <PIL.Image.Image image mode=RGB size=150x150 at 0x1E66E278BB0>
print(type(img))    # <class 'PIL.Image.Image'>
# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape)    # (281, 300, 3) -> (150, 150, 3)
print(type(arr))    # <class 'numpy.ndarray'>

# 차원증가  // reshape로 해도됨
img = np.expand_dims(arr, axis=0)   #차원을 늘리겠다.
print(img.shape)    # axis 0일때 (1, 150, 150, 3), axis 1일때 (150, 1, 150, 3)

############################# 여기부터 증폭 ################################
datagen = ImageDataGenerator(
                             horizontal_flip=True,
                             vertical_flip=True,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.3,
                             shear_range=25,
                             
                             fill_mode='nearest',
                             
                             )

it = datagen.flow(img,
                  batch_size=1,
                  )

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10,10))   # 여러장 한번에 보겠다.

for i in range(10):
    batch = it.next()
    print(batch.shape)
    image = batch[0].astype('uint8')    # 이것 때문에 리스케일 안쓴다.
    print(image.shape)
    ax[i//5, i%5].imshow(image)     # i//5는 행(row) 위치, i%5는 열(column) 위치
    ax[i//5, i%5].axis('on')       #  눈금 및 라벨을 숨겨서 가시성을 없애는 것을 의미
print(np.min(batch), np.max(batch))

plt.show()










