import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
from sklearn.model_selection import cross_val_score
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], 28,28,1).astype('float32')/255.

# print(x_test.shape)

#2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=64, node2=64, node3=64, lr=0.001):
    inputs = Input(shape=(28, 28, 1), name='inputs')
    x = Conv2D(node1, 2, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(node2, 2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Conv2D(node1, 2, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Conv2D(node3, 2, activation=activation, name='hidden4')(x)
    x = Dropout(drop)(x)
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)

    if optimizer == 'adam':
        opt = Adam(learning_rate=lr)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=lr)
    elif optimizer == 'adadelta':
        opt = Adadelta(learning_rate=lr)
    else:
        raise ValueError('Invalid optimizer specified.')

    model.compile(optimizer=opt, metrics=['acc'],
                  loss='sparse_categorical_crossentropy')
    return model

def create_hyperparameter():
    return {
        'batch_size': [100, 200, 300, 400, 500],
        'optimizer': ['adam', 'rmsprop', 'adadelta'],
        'drop': [0.2, 0.3, 0.4, 0.5],
        'activation': ['relu', 'elu', 'selu', 'linear'],
        'lr': [0.1, 0.01, 0.001, 0.0001],
        'node1': [32, 64, 128, 256],
        'node2': [32, 64, 128, 256],
        'node3': [32, 64, 128, 256]
    }

# 콜백 설정
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3)

hyperparameters = create_hyperparameter()
# print(hyperparameter) 
# {'batch_size': [100, 200, 300, 400, 500], 'optimizer': ['adam', 'rmsprop', 'adadlta'], 
# 'dropout': [0.2, 0.3, 0.4, 0.5], 'activation': ['relu', 'elu', 'selu', 'linear']}
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

keras_model = KerasClassifier(build_fn = build_model, verbose=1)
# model = GridSearchCV(keras_moodel,hyperparameters, cv =3)
model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=10, verbose=1)

import time
start = time.time()
model.fit(x_train, y_train, epochs=10, validation_split = 0.2)
end = time.time()

print("걸린시간 : " ,end -start)
print("Best Score: ", model.best_score_) #train데이터의 최고의 스코어
print("Best Params: ", model.best_params_)
print("Best estimator", model.best_estimator_)
print("model Score: ", model.score(x_test,y_test)) #test의 최고 스코어

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('acc :', accuracy_score(y_test,y_predict))

# 걸린시간 :  991.2579295635223
# Best Score:  0.9428499937057495
# Best Params:  {'optimizer': 'adadelta', 'node3': 128, 'node2': 256, 'node1': 64, 'lr': 0.1, 'drop': 0.3, 'batch_size': 300, 'activation': 'selu'}
# Best estimator <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001C19B7B3670>
# 34/34 [==============================] - 0s 10ms/step - loss: 0.0792 - acc: 0.9756
# model Score:  0.975600004196167
# 313/313 [==============================] - 1s 1ms/step
# acc : 0.9756