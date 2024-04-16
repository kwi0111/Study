import numpy as np
from sklearn.datasets import fetch_california_housing 
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, RMSprop, Adadelta

#1. 데이터
datasets = fetch_california_housing()     
x = datasets.data  
y = datasets.target  

x_train, x_test, y_train, y_test = train_test_split(x, y,                     # 훈련 데이터, 테스트 데이터 나누는 과정
                                                    train_size=0.7,
                                                    random_state=123,     
                                                    shuffle=True,
                                                    )

from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = StandardScaler() # 클래스 정의
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
def bulid_model(drop=0.5, optimizer='adam', activation='relu',
                node1=64, node2=32, lr=0.001):
    inputs = Input(shape=(8), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1, name='outputs')(x)

    if optimizer == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr)
    elif optimizer == 'adadelta':
        optimizer = Adadelta(learning_rate=lr)
    
    model = Model(inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['mse'], loss='mae')
    return model


hyperparameters = {
    'drop': [0.2, 0.3, 0.4],
    'optimizer': ['adam', 'rmsprop', 'adadelta'],
    'activation': ['relu', 'elu', 'selu'],
    'node1': [64, 32],
    'node2': [64, 32],
    'lr': [0.001, 0.01, 0.1]
}
# def create_hyperparameter():
#     batchs = [100, 200, 300, 400, 500]
#     optimizers = ['adam', 'rmsprop', 'adadelta']
#     dropouts = [0.2, 0.3, 0.4, 0.5]
#     activations = ['relu', 'elu', 'selu', 'linear']
#     node1 = [128, 64, 32, 16]
#     node2 = [128, 64, 32, 16]
#     node3 = [128, 64, 32, 16]
#     node4 = [128, 64, 32, 16]
#     return {'batch_size' : batchs,
#             'optimizer' : optimizers,
#             'drop' : dropouts,
#             'activation' : activations,
#             'node1' : node1,
#             'node2' : node2,
#             'node3' : node3,
#             'node4' : node4,
#             }
    
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)

import time
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
keras_model = KerasClassifier(build_fn=bulid_model, verbose=1) # scikit learn -> keras

model = RandomizedSearchCV(keras_model, hyperparameters, cv=5, n_iter=10, n_jobs=1, verbose=1)

start_time = time.time()
model.fit(x_train, y_train, epochs=50)
end_time = time.time()

print('걸린 시간 : ', round(end_time-start_time, 2))
print('model.best_params_ : ', model.best_params_)
print('model.best_estimator_ : ', model.best_estimator_)
# 차이 : 디폴트 있냐 없냐 차이
print('model.best_score_ : ', model.best_score_)
# 차이 : 베스트는  train -> 판단은 모델.score

# 평가 메트릭을 MAE로 지정하여 모델을 평가
from sklearn.metrics import mean_absolute_error
y_predict = model.predict(x_test)
mae = mean_absolute_error(y_test, y_predict)
print('Mean Absolute Error (MAE):', mae)


from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('R2 Score:', r2)

# 걸린 시간 :  2279.65
# model.best_params_ :  {'optimizer': 'rmsprop', 'node2': 64, 'node1': 64, 'lr': 0.01, 'drop': 0.3, 'activation': 'elu'}
# model.best_estimator_ :  <keras.wrappers.scikit_learn.KerasClassifier object at 0x0000026215C6FD00>
# model.best_score_ :  nan
# 194/194 [==============================] - 0s 417us/step
# Mean Absolute Error (MAE): 1.8280053076053608
# 194/194 [==============================] - 0s 391us/step
# R2 Score: -2.527038615221222


