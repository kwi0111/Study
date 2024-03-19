from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Concatenate, Add, Activation
import tensorflow as tf

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

def MultiResUNet(channels, filters=32, nclasses=1):
    alpha = 1.67

    inputs = Input(shape=(None, None, channels))
    
    multires1 = multiresblock(inputs, channels, filters)
    pool1 = MaxPooling2D((2, 2))(multires1)
    respath1 = respath(multires1, channels, filters, 4)
    
    multires2 = multiresblock(pool1, filters, filters*2)
    pool2 = MaxPooling2D((2, 2))(multires2)
    respath2 = respath(multires2, filters, filters*2, 3)
    
    multires3 = multiresblock(pool2, filters*2, filters*4)
    pool3 = MaxPooling2D((2, 2))(multires3)
    respath3 = respath(multires3, filters*2, filters*4, 2)
    
    multires4 = multiresblock(pool3, filters*4, filters*8)
    pool4 = MaxPooling2D((2, 2))(multires4)
    respath4 = respath(multires4, filters*4, filters*8, 1)
    
    multires5 = multiresblock(pool4, filters*8, filters*16)
    
    up6 = Concatenate()([UpSampling2D(size=(2, 2))(multires5), multires4])
    multires6 = multiresblock(up6, filters*16, filters*8)
    
    up7 = Concatenate()([UpSampling2D(size=(2, 2))(multires6), multires3])
    multires7 = multiresblock(up7, filters*8, filters*4)
    
    up8 = Concatenate()([UpSampling2D(size=(2, 2))(multires7), multires2])
    multires8 = multiresblock(up8, filters*4, filters*2)
    
    up9 = Concatenate()([UpSampling2D(size=(2, 2))(multires8), multires1])
    multires9 = multiresblock(up9, filters*2, filters)
    
    if nclasses > 1:
        outputs = Conv2D(nclasses, (1, 1), activation='softmax')(multires9)
    else:
        outputs = Activation('sigmoid')(Conv2D(1, (1, 1))(multires9))

    model = Model(inputs, outputs)
    return model

model = MultiResUNet(channels=3, filters=32, nclasses=10)
model.summary()
