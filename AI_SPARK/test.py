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
                

# 모델 정의

def conv2d_batchnorm(x, filters, kernel_size=(2, 2), activation='relu', padding='same'):
    x = Conv2D(filters, kernel_size, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == 'relu':
        x = Activation('relu')(x)
    return x

def multiresblock(x, input_features, corresponding_unet_filters, alpha=1.67):
    W = corresponding_unet_filters * alpha

    temp = conv2d_batchnorm(x, int(W*0.167) + int(W*0.333) + int(W*0.5), kernel_size=(1, 1), activation=None)
    a = conv2d_batchnorm(x, int(W*0.167), kernel_size=(3, 3), activation='relu')
    b = conv2d_batchnorm(a, int(W*0.333), kernel_size=(3, 3), activation='relu')
    c = conv2d_batchnorm(b, int(W*0.5), kernel_size=(3, 3), activation='relu')
    x = Concatenate(axis=-1)([a, b, c])
    x = BatchNormalization()(x)
    x = Add()([temp, x])
    x = BatchNormalization()(x)
    return x

def respath(x, input_features, filters, respath_length):
    shortcut = conv2d_batchnorm(x, filters, kernel_size=(1, 1), activation=None)
    x = conv2d_batchnorm(x, filters, kernel_size=(3, 3), activation='relu')
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    for _ in range(respath_length):
        shortcut = conv2d_batchnorm(x, filters, kernel_size=(1, 1), activation=None)
        x = conv2d_batchnorm(x, filters, kernel_size=(3, 3), activation='relu')
        x = Add()([shortcut, x])
        x = Activation('relu')(x)
    return x

def MultiResUNet_model(input_shape=(None, None, 3), filters=32, nclasses=1):
    inputs = Input(shape=input_shape)
    alpha = 1.67
    
    # Encoding Path
    multires1 = multiresblock(inputs, input_shape[-1], filters, alpha)
    pool1 = MaxPooling2D(pool_size=(2, 2))(multires1)
    respath1 = respath(multires1, input_shape[-1], filters, 4)
    
    multires2 = multiresblock(pool1, filters, filters*2, alpha)
    pool2 = MaxPooling2D(pool_size=(2, 2))(multires2)
    respath2 = respath(multires2, filters, filters*2, 3)
    
    multires3 = multiresblock(pool2, filters*2, filters*4, alpha)
    pool3 = MaxPooling2D(pool_size=(2, 2))(multires3)
    respath3 = respath(multires3, filters*2, filters*4, 2)
    
    multires4 = multiresblock(pool3, filters*4, filters*8, alpha)
    pool4 = MaxPooling2D(pool_size=(2, 2))(multires4)
    respath4 = respath(multires4, filters*4, filters*8, 1)
    
    # Bridge
    multires5 = multiresblock(pool4, filters*8, filters*16, alpha)
    
    # Decoding Path
    up6 = Concatenate()([UpSampling2D(size=(2, 2))(multires5), respath4])
    multires6 = multiresblock(up6, filters*16, filters*8, alpha)
    
    up7 = Concatenate()([UpSampling2D(size=(2, 2))(multires6), respath3])
    multires7 = multiresblock(up7, filters*8, filters*4, alpha)
    
    up8 = Concatenate()([UpSampling2D(size=(2, 2))(multires7), respath2])
    multires8 = multiresblock(up8, filters*4, filters*2, alpha)
    
    up9 = Concatenate()([UpSampling2D(size=(2, 2))(multires8), respath1])
    multires9 = multiresblock(up9, filters*2, filters, alpha)
    
    # Output Layer
    if nclasses > 1:
        outputs = Conv2D(nclasses, (1, 1), activation='softmax')(multires9)
    else:
        outputs = Activation('sigmoid')(Conv2D(1, (1, 1))(multires9))

    model = Model(inputs, outputs)
    return model




def get_model(model_name, nClasses=1, input_height=128, input_width=128, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=10):
    if model_name == 'MultiResUNet':
        model = MultiResUNet_model

    return model(
            # nClasses      = nClasses,  
            # input_height  = input_height, 
            # input_width   = input_width,
            # n_filters     = n_filters,
            # dropout       = dropout,
            # batchnorm     = batchnorm,
            # n_channels    = n_channels
        )
    

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
EPOCHS = 200 # 훈련 epoch 지정
BATCH_SIZE = 6 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'MultiResUNet' # 모델 이름
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

learning_rate = 0.0001
model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, )
model.compile(optimizer = Adam(learning_rate=learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

model.load_weights('D:\\_data\\dataset\\output\\model_MultiResUNet_sample_line_final_weights.h5')

y_pred_dict = {}

for i in test_meta['test_img']:
    img = get_img_762bands(f'D:\\_data\\dataset\\test_img\\{i}')
    y_pred = model.predict(np.array([img]), batch_size=1, verbose=0)
    
    y_pred = np.where(y_pred[0, :, :, 0] > 0.25, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred

joblib.dump(y_pred_dict, 'D:\\_data\\dataset\\output\\y_pred.pkl')