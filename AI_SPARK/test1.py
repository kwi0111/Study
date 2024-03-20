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







from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, Concatenate

def conv_block(x, filters, kernel_size=(3, 3), strides=(1, 1), padding='same'):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def convnext_block(x, filters, cardinality=32, strides=(1, 1), padding='same'):
    grouped_channels = filters // cardinality
    groups = []

    for i in range(cardinality):
        group = conv_block(x, grouped_channels, strides=strides, padding=padding)
        groups.append(group)

    x = Concatenate()(groups)
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def bottleneck_block(x, filters, cardinality=32, strides=(1, 1), padding='same'):
    shortcut = x

    x = convnext_block(x, filters, cardinality, strides, padding)
    x = conv_block(x, filters, kernel_size=(1, 1), padding='valid')

    if strides != (1, 1) or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def ConvNeXt(input_shape, num_classes, base_filters=64, cardinality=32):
    inputs = Input(shape=input_shape)

    x = conv_block(inputs, base_filters)
    x = MaxPooling2D((2, 2))(x)

    x = bottleneck_block(x, base_filters * 2, cardinality, strides=(2, 2))
    x = bottleneck_block(x, base_filters * 2, cardinality)

    x = bottleneck_block(x, base_filters * 4, cardinality, strides=(2, 2))
    x = bottleneck_block(x, base_filters * 4, cardinality)

    x = bottleneck_block(x, base_filters * 8, cardinality, strides=(2, 2))
    x = bottleneck_block(x, base_filters * 8, cardinality)

    x = MaxPooling2D((2, 2))(x)
    x = conv_block(x, base_filters * 16)
    
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# 모델 생성
input_shape = (224, 224, 3)  # 예시로 입력 이미지 크기는 (224, 224, 3)으로 설정
num_classes = 2  # 이진 분류 예시
model = ConvNeXt(input_shape, num_classes)

# 모델 요약 출력
model.summary()






from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate

def conv_block(inputs, filters, kernel_size=(3, 3), activation='relu', padding='same'):
    conv = Conv2D(filters, kernel_size, activation=activation, padding=padding)(inputs)
    conv = Conv2D(filters, kernel_size, activation=activation, padding=padding)(conv)
    return conv

def unet(input_shape, num_classes):
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_block(pool3, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = conv_block(pool4, 1024)
    
    # Decoder
    up6 = UpSampling2D(size=(2, 2))(conv5)
    concat6 = concatenate([conv4, up6], axis=3)
    conv6 = conv_block(concat6, 512)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    concat7 = concatenate([conv3, up7], axis=3)
    conv7 = conv_block(concat7, 256)
    
    up8 = UpSampling2D(size=(2, 2))(conv7)
    concat8 = concatenate([conv2, up8], axis=3)
    conv8 = conv_block(concat8, 128)
    
    up9 = UpSampling2D(size=(2, 2))(conv8)
    concat9 = concatenate([conv1, up9], axis=3)
    conv9 = conv_block(concat9, 64)
    
    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 모델 생성
input_shape = (256, 256, 3)  # 입력 이미지 크기
num_classes = 1  # 출력 클래스 수 (이진 분할)
model = unet(input_shape, num_classes)

# 모델 요약 출력
model.summary()