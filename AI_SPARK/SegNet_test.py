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

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization, Activation, Add
from keras.optimizers import Adam

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
def generator_from_lists(images_path, masks_path, batch_size=32, shuffle=True, random_state=None, image_mode='10bands'):
   
    images = []
    masks = []

    fopen_image = get_img_arr
    fopen_mask = get_mask_arr

    if image_mode == '762':
        fopen_image = get_img_762bands

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
            
            # 여기에 차원 확인 코드 추가
            # print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")

            images.append(img)
            masks.append(mask)

            if len(images) >= batch_size:
                yield (np.array(images), np.array(masks))
                images = []
                masks = []
                

# Unet 모델 정의
def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_unet(nClasses, input_height=256, input_width=256, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=3):
    input_img = Input(shape=(input_height,input_width, n_channels))

    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
    
def conv_block(input_tensor, n_filters, kernel_size, batchnorm):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    return x

def residual_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Residual block with two convolutional layers and a shortcut connection."""
    x = conv_block(input_tensor, n_filters, kernel_size, batchnorm)
    shortcut = Conv2D(n_filters, kernel_size=(1, 1), kernel_initializer="he_normal", padding="same")(input_tensor)
    if batchnorm:
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    return x

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization, Activation, Multiply, Add
from keras.models import Model

def attention_block(F_g, F_l, inter_channel):
    W_g = Conv2D(inter_channel, kernel_size=1, padding='same')(F_g)
    W_g = BatchNormalization()(W_g)
    W_g = Activation('relu')(W_g)

    W_l = Conv2D(inter_channel, kernel_size=1, padding='same')(F_l)
    W_l = BatchNormalization()(W_l)
    W_l = Activation('relu')(W_l)

    psi = Add()([W_g, W_l])
    psi = Conv2D(1, kernel_size=1, padding='same')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)

    return Multiply()([F_l, psi])


def segnet(input_size=(256, 256, 3), n_classes=1, n_filters=64, dropout=0.1):
    inputs = Input(input_size)
    
    # Encoder
    c1 = residual_block(inputs, n_filters * 1, kernel_size=3, batchnorm=True)
    p1 = MaxPooling2D((2, 2))(c1)
    d1 = Dropout(dropout)(p1)
    
    c2 = residual_block(d1, n_filters * 2, kernel_size=3, batchnorm=True)
    p2 = MaxPooling2D((2, 2))(c2)
    d2 = Dropout(dropout)(p2)
    
    c3 = residual_block(d2, n_filters * 4, kernel_size=3, batchnorm=True)
    p3 = MaxPooling2D((2, 2))(c3)
    d3 = Dropout(dropout)(p3)
    
    c4 = residual_block(d3, n_filters * 8, kernel_size=3, batchnorm=True)
    p4 = MaxPooling2D((2, 2))(c4)
    d4 = Dropout(dropout)(p4)
    
    # Bridge
    c5 = conv_block(d4, n_filters * 16, kernel_size=3, batchnorm=True)
    
    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    a6 = attention_block(u6, c4, n_filters * 8)
    u6 = concatenate([u6, a6])
    c6 = conv_block(u6, n_filters * 8, kernel_size=3, batchnorm=True)
    
    u7 = UpSampling2D((2, 2))(c6)
    a7 = attention_block(u7, c3, n_filters * 4)
    u7 = concatenate([u7, a7])
    c7 = conv_block(u7, n_filters * 4, kernel_size=3, batchnorm=True)
    
    u8 = UpSampling2D((2, 2))(c7)
    a8 = attention_block(u8, c2, n_filters * 2)
    u8 = concatenate([u8, a8])
    c8 = conv_block(u8, n_filters * 2, kernel_size=2, batchnorm=True)
    
    u9 = UpSampling2D((2, 2))(c8)
    a9 = attention_block(u9, c1, n_filters)
    u9 = concatenate([u9, a9])
    c9 = conv_block(u9, n_filters, kernel_size=2, batchnorm=True)
    
    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(inputs)  # 실제 모델에서는 마지막 디코더 레이어를 사용해야 함
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def get_model(model_name, input_height=128, input_width=128, n_filters=16, dropout=0.05, batchnorm=True, n_channels=3):
    if model_name == 'unet':
        model = get_unet(nClasses=1, input_height=input_height, input_width=input_width, n_filters=n_filters, dropout=dropout, batchnorm=batchnorm, n_channels=n_channels)
    elif model_name == 'segnet':
        model = segnet(input_size=(input_height, input_width, n_channels), n_classes=1, n_filters=n_filters, dropout=dropout)

    return model
    

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
EPOCHS = 30 # 훈련 epoch 지정
BATCH_SIZE = 24 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'segnet' # 모델 이름
RANDOM_STATE = 42 # seed 고정
INITIAL_EPOCH = 0 # 초기 epoch

# 데이터 위치
IMAGES_PATH = 'D:\\_data\\dataset\\train_img\\'
MASKS_PATH = 'D:\\_data\\dataset\\train_mask\\'

# 가중치 저장 위치
OUTPUT_DIR = 'D:\\_data\\dataset\\output\\'
WORKERS = 22

# 조기종료
EARLY_STOP_PATIENCE = 5 

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
learning_rate = 0.01
model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS,)
model.compile(optimizer = Adam(learning_rate=learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy', miou])
# model.summary()

# checkpoint 및 조기종료 설정
es = EarlyStopping(monitor='val_miou', mode='max', verbose=1, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='val_miou', verbose=1,
save_best_only=True, mode='max', period=CHECKPOINT_PERIOD)



print('---model 훈련 시작---')
history = model.fit_generator(
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

print('가중치 저장')
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("저장된 가중치 명: {}".format(model_weights_output))

learning_rate = 0.01
# model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, )
model = segnet(input_size=(256, 256, 3), n_classes=1, n_filters=64, dropout=0.1)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.load_weights('D:\\_data\\dataset\\output\\model_segnet_sample_line_final_weights.h5')

y_pred_dict = {}

for i in test_meta['test_img']:
    img = get_img_762bands(f'D:\\_data\\dataset\\test_img\\{i}')
    y_pred = model.predict(np.array([img]), batch_size=1, verbose=0)
    
    y_pred = np.where(y_pred[0, :, :, 0] > 0.25, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred

joblib.dump(y_pred_dict, 'D:\\_data\\dataset\\output\\y_pred.pkl')
