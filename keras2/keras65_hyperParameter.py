import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.0
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.0


#2. 모델
def bulid_model(drop = 0.5, optimizer = 'adam', activation = 'relu',
                node1=128, node2=64, node3=32, lr = 0.001):
    inputs = Input(shape=(28*28), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    
    model = Model(inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'], loss='sparse_categorical_crossentropy')
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
            'drop' : dropouts,
            'activation' : activations,
            'node1' : node1,
            'node2' : node2,
            'node3' : node3,
            }
    
hyperparameters = create_hyperparameter()
print(hyperparameters)

import time
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
keras_model = KerasClassifier(build_fn=bulid_model, verbose=1) # scikit learn -> keras

# model = RandomizedSearchCV(bulid_model, hyperparameters, cv=2, n_iter=10, n_jobs=-1, verbose=1)
# 랜덤서치가 못알아먹는 모델, 파라미터 사용으로 에러 -> 케라스 모델을 사이킷런 머신러닝 모델로 바꿔줘야함.
model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=1, n_jobs=-1, verbose=1)


start_time = time.time()
model.fit(x_train, y_train, epochs=3)
end_time = time.time()

print('걸린 시간 : ', round(end_time-start_time, 2))
print('model.best_params_ : ', model.best_params_)
print('model.best_estimator_ : ', model.best_estimator_)
# 차이 : 디폴트 있냐 없냐 차이
print('model.best_score_ : ', model.best_score_)
print('model.score_ : ', model.score(x_test, y_test))
# 차이 : 베스트는  train -> 판단은 모델.score

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('acc_score : ', accuracy_score(y_test, y_predict))

# 걸린 시간 :  9.53
# model.best_params_ :  {'optimizer': 'rmsprop', 'node3': 64, 'node2': 16, 'node1': 16, 'drop': 0.3, 'batch_size': 200, 'activation': 'linear'}
# model.best_estimator_ :  <keras.wrappers.scikit_learn.KerasClassifier object at 0x000002043155E3D0>
# model.best_score_ :  0.8866666555404663
# 50/50 [==============================] - 0s 1ms/step - loss: 0.3836 - acc: 0.8922
# model.score_ :  0.8921999931335449
# 313/313 [==============================] - 0s 519us/step
# acc_score :  0.8922
