import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

#CNN
# 얼리 스탑핑 적용
# MCP 적용


#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    random_state=123,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    )
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
scaler = StandardScaler() # 클래스 정의

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train = x_train.reshape(-1, 30)
x_test = x_test.reshape(-1, 30)

#2. 모델
def bulid_model(drop = 0.5, optimizer = 'adam', activation = 'relu',
                node1=128, node2=64, node3=32, lr = 0.001):
    inputs = Input(shape=(30,), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1, activation='sigmoid', name='outputs')(x)
    
    model = Model(inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'], loss='binary_crossentropy')
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
model = RandomizedSearchCV(keras_model, hyperparameters, cv=3, n_iter=10, n_jobs=-1, verbose=1)


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

# 걸린 시간 :  26.2
# model.best_params_ :  {'optimizer': 'adam', 'node3': 128, 'node2': 128, 'node1': 128, 'drop': 0.4, 'batch_size': 500, 'activation': 'linear'}
# model.best_estimator_ :  <keras.wrappers.scikit_learn.KerasClassifier object at 0x0000028F883F16A0>
# model.best_score_ :  0.9252933661142985
# 1/1 [==============================] - 0s 66ms/step - loss: 0.1686 - acc: 0.9561
# model.score_ :  0.9561403393745422
# 4/4 [==============================] - 0s 0s/step
# acc_score :  0.956140350877193
