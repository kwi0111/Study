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
img = np.expand_dims(arr, axis=1)
print(img.shape)    # axis 0일때 (1, 150, 150, 3), axis 1일때 (150, 1, 150, 3)










