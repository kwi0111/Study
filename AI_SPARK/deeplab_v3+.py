# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings("ignore")
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras import backend as K
import sys
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import threading
import random
import rasterio
import os
import numpy as np
import sys
from sklearn.utils import shuffle as shuffle_lists
from keras.models import *
from keras.layers import *
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split
import joblib

np.random.seed(99)       # 0
random.seed(1)         # 42 
tf.random.set_seed(19)   # 7

MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값

class threadsafe_iter:
    """
    데이터 불러올떼, 호출 직렬화
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g

def get_img_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    img = np.float32(img)/MAX_PIXEL_VALUE

    return img

def get_img_762bands(path):
    img = rasterio.open(path).read((7,6,2)).transpose((1, 2, 0))
    img = np.float32(img)/MAX_PIXEL_VALUE

    return img

def get_mask_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg



@threadsafe_generator
def generator_from_lists(images_path, masks_path, batch_size=32, shuffle = True, random_state=None, image_mode='10bands'):

    images = []
    masks = []

    fopen_image = get_img_arr
    fopen_mask = get_mask_arr

    if image_mode == '762':
        fopen_image = get_img_762bands

    i = 0
    # 데이터 shuffle
    while True:

        if shuffle:
            if random_state is None:
                images_path, masks_path = shuffle_lists(images_path, masks_path)
            else:
                images_path, masks_path = shuffle_lists(images_path, masks_path, random_state= random_state + i)
                i += 1


        for img_path, mask_path in zip(images_path, masks_path):

            img = fopen_image(img_path)
            mask = fopen_mask(mask_path)
            images.append(img)
            masks.append(mask)

            if len(images) >= batch_size:
                yield (np.array(images), np.array(masks))
                images = []
                masks = []

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate
from tensorflow.keras.applications import Xception

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, concatenate, Input

class threadsafe_iter:
    """데이터 불러올 때, 호출을 직렬화합니다."""
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """스레드 안전한 생성자를 만들기 위한 데코레이터입니다."""
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def get_img_arr(path):
    """이미지 파일을 numpy 배열로 로드합니다."""
    img = rasterio.open(path).read().transpose((1, 2, 0))
    img = np.float32(img) / MAX_PIXEL_VALUE
    return img

def get_img_762bands(path):
    """특정 밴드를 사용하여 이미지 파일을 numpy 배열로 로드합니다."""
    img = rasterio.open(path).read((7, 6, 2)).transpose((1, 2, 0))
    img = np.float32(img) / MAX_PIXEL_VALUE
    return img

def get_mask_arr(path):
    """마스크 이미지 파일을 numpy 배열로 로드합니다."""
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg

@threadsafe_generator
def generator_from_lists(images_path, masks_path, batch_size=32, shuffle=True, random_state=None, image_mode='10bands'):
    """이미지와 마스크 경로 목록에서 배치 데이터를 생성하는 생성자입니다."""
    images = []
    masks = []
    fopen_image = get_img_arr if image_mode == '10bands' else get_img_762bands
    fopen_mask = get_mask_arr

    i = 0
    while True:
        if shuffle:
            if random_state is None:
                images_path, masks_path = shuffle_lists(images_path, masks_path)
            else:
                images_path, masks_path = shuffle_lists(images_path, masks_path, random_state=random_state + i)
                i += 1

        for img_path, mask_path in zip(images_path, masks_path):
            img = fopen_image(img_path)
            mask = fopen_mask(mask_path)
            images.append(img)
            masks.append(mask)

            if len(images) >= batch_size:
                yield (np.array(images), np.array(masks))
                images = []
                masks = []

# MobileNetV2를 백본으로 하는 U-Net 아키텍처를 정의합니다.
def MobileNetV2_UNet(input_height=128, input_width=128, n_channels=3, n_classes=1):
    input_shape = (input_height, input_width, n_channels)
    inputs = Input(input_shape)
    encoder = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)

    # Encoder에서 skip connection을 얻습니다.
    s1 = encoder.get_layer('block_1_expand_relu').output
    s2 = encoder.get_layer('block_3_expand_relu').output
    s3 = encoder.get_layer('block_6_expand_relu').output
    s4 = encoder.get_layer('block_13_expand_relu').output
    b1 = encoder.get_layer('out_relu').output  # Bridge

    # Decoder
    d1 = Conv2DTranspose(512, (3, 3), strides=(1, 1), padding='same')(b1)
    d1 = Concatenate()([d1, s4])  # s4와 concatenate
    d1 = Conv2D(512, (3, 3), activation='relu', padding='same')(d1)

    d2 = Conv2DTranspose(256, (3, 3), strides=(1, 1), padding='same')(d1)
    d2 = Concatenate()([d2, s3])  # s3와 concatenate
    d2 = Conv2D(256, (3, 3), activation='relu', padding='same')(d2)

    d3 = Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same')(d2)
    d3 = Concatenate()([d3, s2])  # s2와 concatenate
    d3 = Conv2D(128, (3, 3), activation='relu', padding='same')(d3)

    d4 = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same')(d3)
    d4 = Concatenate()([d4, s1])  # s1와 concatenate
    d4 = Conv2D(64, (3, 3), activation='relu', padding='same')(d4)

    d5 = Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same')(d4)
    d5 = Conv2D(32, (3, 3), activation='relu', padding='same')(d5)

    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(d5)

    model = Model(inputs, outputs)
    return model

# 모델 선택 함수를 정의합니다.
def get_model(model_name, input_height=64, input_width=64, n_channels=3, n_classes=1):
    """지정된 모델 이름에 따라 모델을 반환합니다."""
    if model_name == 'MobileNetV2_UNet':
        return MobileNetV2_UNet(input_height, input_width, n_channels, n_classes)
    else:
        raise ValueError("Unsupported model type. Please check the model name and try again.")



"""&nbsp; 

## parameter 설정
"""

# 사용할 데이터의 meta정보 가져오기

train_meta = pd.read_csv('d:/_data/dataset/train_meta.csv')
test_meta = pd.read_csv('d:/_data/dataset/test_meta.csv')


# 저장 이름
save_name = 'base_line'

N_FILTERS = 8 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 1 # 훈련 epoch 지정
BATCH_SIZE = 4 # batch size 지정
IMAGE_SIZE = (32, 32) # 이미지 크기 지정
MODEL_NAME = 'MobileNetV2_UNet' # 모델 이름
RANDOM_STATE = 42 # seed 고정  // 원래 47
INITIAL_EPOCH = 0 # 초기 epoch

# 데이터 위치
IMAGES_PATH = 'D:\\_data\\dataset\\train_img\\'
MASKS_PATH = 'D:\\_data\\dataset\\train_mask\\'

# 가중치 저장 위치
OUTPUT_DIR = 'D:\\_data\\dataset\\output\\'
WORKERS = 8    # 원래 4 // (코어 / 2 ~ 코어) 

# 조기종료
EARLY_STOP_PATIENCE = 20

# 중간 가중치 저장 이름
CHECKPOINT_PERIOD = 5
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}.hdf5'.format(MODEL_NAME, save_name)

# 최종 가중치 저장 이름
FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_final_weights.h5'.format(MODEL_NAME, save_name)

# 사용할 GPU 이름
CUDA_DEVICE = 0


# 저장 폴더 없으면 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
except:
    pass

try:
    np.random.bit_generator = np.random._bit_generator
except:
    pass


# train : val = 8 : 2 나누기
x_tr, x_val = train_test_split(train_meta, test_size=0.2, random_state=RANDOM_STATE)
print(len(x_tr), len(x_val))

# train : val 지정 및 generator
images_train = [os.path.join(IMAGES_PATH, image) for image in x_tr['train_img'] ]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in x_tr['train_mask'] ]

images_validation = [os.path.join(IMAGES_PATH, image) for image in x_val['train_img'] ]
masks_validation = [os.path.join(MASKS_PATH, mask) for mask in x_val['train_mask'] ]

train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")
validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")



#miou metric
Threshold  = 0.75
def miou(y_true, y_pred, smooth=1e-6):
    # 임계치 기준으로 이진화
    y_pred = tf.cast(y_pred > Threshold , tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    
    # mIoU 계산
    iou = (intersection + smooth) / (union + smooth)
    miou = tf.reduce_mean(iou)
    return miou


# model 불러오기
learning_rate = 0.001
# model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_channels=N_CHANNELS)
model = get_model(MODEL_NAME, IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS, 1)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', miou])



# checkpoint 및 조기종료 설정
es = EarlyStopping(monitor='val_miou', mode='max', verbose=1, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='val_miou', verbose=1,
save_best_only=True, mode='max', period=CHECKPOINT_PERIOD)

# rlr
# rlr = ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=0, mode='auto', factor=0.5)


"""&nbsp;

## model 훈련
"""

print('---model 훈련 시작---')
history = model.fit(
    train_generator,
    steps_per_epoch=len(images_train) // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(images_validation) // BATCH_SIZE,
    callbacks=[checkpoint, es],
    epochs=EPOCHS,
    workers=WORKERS,
    initial_epoch=INITIAL_EPOCH
)
print('---model 훈련 종료---')

"""&nbsp;

## model save
"""

print('가중치 저장')
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("저장된 가중치 명: {}".format(model_weights_output))

"""## inference

- 학습한 모델 불러오기
"""
learning_rate = 0.001
# model = get_model(MODEL_NAME, input_height=128, input_width=128, n_filters=N_FILTERS, n_channels=N_CHANNELS)
model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
# model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.compile(optimizer = Adam(learning_rate=learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy', miou])
# model.summary()

model.load_weights('D:\\_data\\dataset\\output\\model_unet_base_line_final_weights.h5')

"""## 제출 Predict
- numpy astype uint8로 지정
- 반드시 pkl로 저장

"""

y_pred_dict = {}

for i in test_meta['test_img']:
    img = get_img_762bands(f'd:/_data/dataset/test_img/{i}')
    y_pred = model.predict(np.array([img]), batch_size=1)

    y_pred = np.where(y_pred[0, :, :, 0] > 0.5, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred

from datetime import datetime
dt = datetime.now()
joblib.dump(y_pred_dict, f'D:\\_data\\dataset\\output\\y_pred_{dt.day}_{dt.hour}_{dt.minute}.pkl')

# https://aifactory.space/task/2723/submit