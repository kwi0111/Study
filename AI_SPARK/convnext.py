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
from keras.callbacks import ModelCheckpoint, EarlyStopping
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
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from datetime import datetime


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
                


############################## 모델 정의 #################################
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


class StochasticDepth(layers.Layer):
    """Stochastic Depth module.

    It is also referred to as Drop Path in `timm`.
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class Block(tf.keras.Model):
    """ConvNeXt block.

    References:
        (1) https://arxiv.org/abs/2201.03545
        (2) https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6, **kwargs):
        super(Block, self).__init__(**kwargs)
        self.dim = dim
        if layer_scale_init_value > 0:
            self.gamma = tf.Variable(layer_scale_init_value * tf.ones((dim,)))
        else:
            self.gamma = None
        self.dw_conv_1 = layers.Conv2D(
            filters=dim, kernel_size=7, padding="same", groups=dim
        )
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.pw_conv_1 = layers.Dense(4 * dim)
        self.act_fn = layers.Activation("gelu")
        self.pw_conv_2 = layers.Dense(dim)
        self.drop_path = (
            StochasticDepth(drop_path)
            if drop_path > 0.0
            else layers.Activation("linear")
        )

    def call(self, input_1):
        x = input_1

        x = self.dw_conv_1(x)
        x = self.layer_norm(x)
        x = self.pw_conv_1(x)
        x = self.act_fn(x)
        x = self.pw_conv_2(x)

        if self.gamma is not None:
            x = self.gamma * x

        return input_1 + self.drop_path(x)


def get_convnext_model(
    model_name,
    input_1_shape,
    num_classes,
    depths,
    dims,
    drop_path_rate,
    layer_scale_init_value,
) -> keras.Model:
    """Implements ConvNeXt family of models given a configuration.

    References:
        (1) https://arxiv.org/abs/2201.03545
        (2) https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Note: `predict()` fails on CPUs because of group convolutions. The fix is recent at
    the time of the development: https://github.com/keras-team/keras/pull/15868. It's
    recommended to use a GPU / TPU.
    """

    input_1 = layers.Input(input_1_shape, name="input_1")
    stem = keras.Sequential(
        [
            layers.Conv2D(dims[0], kernel_size=4, strides=4),
            layers.LayerNormalization(epsilon=1e-6),
        ],
        name="stem",
    )

    downsample_layers = []
    downsample_layers.append(stem)
    for i in range(3):
        downsample_layer = keras.Sequential(
            [
                layers.LayerNormalization(epsilon=1e-6),
                layers.Conv2D(dims[i + 1], kernel_size=2, strides=2),
            ],
            name=f"downsampling_block_{i}",
        )
        downsample_layers.append(downsample_layer)

    stages = []
    dp_rates = [x for x in tf.linspace(0.0, drop_path_rate, sum(depths))]
    cur = 0
    for i in range(4):
        stage = keras.Sequential(
            [
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        name=f"convnext_block_{i}_{j}",
                    )
                    for j in range(depths[i])
                ]
            ],
            name=f"convnext_stage_{i}",
        )
        stages.append(stage)
        cur += depths[i]

    x = input_1
    for i in range(len(stages)):
        x = downsample_layers[i](x)
        x = stages[i](x)

    # 업샘플링 시작
    x = layers.Conv2DTranspose(filters=384, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=384, kernel_size=3, padding='same')(x)  # 추가된 Conv2D
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(filters=192, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=192, kernel_size=3, padding='same')(x)  # 추가된 Conv2D
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(filters=96, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=96, kernel_size=3, padding='same')(x)  # 추가된 Conv2D
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(filters=48, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=48, kernel_size=3, padding='same')(x)  # 추가된 Conv2D
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('sigmoid')(x)  # 마지막 활성화 함수는 목적에 따라 달라질 수 있음

    return keras.Model(inputs=input_1, outputs=x, name=model_name)


# 두 샘플 간의 유사성 metric
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

# 픽셀 정확도를 계산 metric
def pixel_accuracy (y_true, y_pred):
    sum_n = np.sum(np.logical_and(y_pred, y_true))
    sum_t = np.sum(y_true)
 
    if (sum_t == 0):
        pixel_accuracy = 0
    else:
        pixel_accuracy = sum_n / sum_t
    return pixel_accuracy    


# 사용할 데이터의 meta정보 가져오기

train_meta = pd.read_csv('D:\\_data\\dataset\\train_meta.csv')
test_meta = pd.read_csv('D:\\_data\\dataset\\test_meta.csv')


# 저장 이름
save_name = 'sample_line'

N_FILTERS = 16 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 100 # 훈련 epoch 지정
BATCH_SIZE = 16 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'get_convnext_model' # 모델 이름
RANDOM_STATE = 42 # seed 고정
INITIAL_EPOCH = 0 # 초기 epoch

# 데이터 위치
IMAGES_PATH = 'D:\\_data\\dataset\\train_img\\'
MASKS_PATH = 'D:\\_data\\dataset\\train_mask\\'

# 가중치 저장 위치
OUTPUT_DIR = 'D:\\_data\\dataset\\output\\'
WORKERS = 22

# 조기종료
EARLY_STOP_PATIENCE = 15

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
Threshold  = 0.5
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
learning_rate = 0.0001
model = get_convnext_model(
    model_name="get_convnext_model",
    input_1_shape=(256, 256, 3),  # 예시 입력 크기로 설정
    num_classes=1,  # 이진 분류를 가정
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    drop_path_rate=0.0,
    layer_scale_init_value=1e-6,
)

model.compile(optimizer = Adam(learning_rate=learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy', miou])
model.summary()

# checkpoint 및 조기종료 설정
checkpoint_dir = 'D:/_data/dataset/output/model_checkpoints/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_filepath = os.path.join(checkpoint_dir, f"model_{timestamp}_{{epoch:02d}}_{{val_miou:.4f}}.hdf5")

# es = EarlyStopping(monitor='val_miou', mode='max', verbose=1, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)

# checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='val_miou', verbose=1,
# save_best_only=True, mode='max', period=CHECKPOINT_PERIOD)
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_miou',
    verbose=1,
    save_best_only=True,
    mode='max',
    save_weights_only=True  # 모델의 가중치만 저장
)

es = EarlyStopping(
    monitor='val_miou',
    mode='max',
    verbose=1,
    patience=EARLY_STOP_PATIENCE,
    restore_best_weights=True
)

rlr = ReduceLROnPlateau(monitor='val_miou', patience=10, factor=0.5, mode='max')

print('---model 훈련 시작---')
history = model.fit(
    train_generator,
    steps_per_epoch=len(images_train) // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(images_validation) // BATCH_SIZE,
    callbacks=[checkpoint_callback, es, rlr],
    epochs=EPOCHS,
    workers=WORKERS,
    initial_epoch=INITIAL_EPOCH
)
print('---model 훈련 종료---')


print('가중치 저장')
from datetime import datetime
# 현재 날짜와 시간을 포함한 고유한 파일 이름 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = os.path.join(OUTPUT_DIR, f"model_{timestamp}")

# 변경된 파일 이름으로 가중치 저장
model.save(model_save_path)
print(f"모델이 저장된 경로: {model_save_path}")



# learning_rate = 0.00001
# model = get_convnext_model(
#     model_name="get_convnext_model",
#     input_1_shape=(256, 256, 3),  # 예시 입력 크기로 설정
#     num_classes=1,  # 이진 분류를 가정
#     depths=[3, 3, 9, 3],
#     dims=[96, 192, 384, 768],
#     drop_path_rate=0.0,
#     layer_scale_init_value=1e-6,
# )
# model.compile(optimizer = Adam(learning_rate=learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy', miou])
# model.summary()

# from keras.models import load_model
# model_load_path = 'D:\\_data\\dataset\\output\\model_YYYYMMDD_HHMMSS'
# loaded_model = load_model(model_load_path)
# model.load_weights('D:\\_data\\dataset\\output\\model_get_convnext_model_sample_line_final_weights.h5', by_name=True)


y_pred_dict = {}

for i in test_meta['test_img']:
    img = get_img_762bands(f'D:\\_data\\dataset\\test_img\\{i}')
    y_pred = model.predict(np.array([img]), batch_size=1, verbose=1)
    
    y_pred = np.where(y_pred[0, :, :, 0] > 0.25, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred

dt = datetime.now()
joblib.dump(y_pred_dict, f'D:\\_data\\dataset\\output\\y_pred_{dt.day}_{dt.hour}_{dt.minute}.pkl')
print(f'끝. : y_pred_{dt.day}_{dt.hour}_{dt.minute}.pkl ')

