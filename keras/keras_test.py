import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

path = 'asda//sd'
train_csv = pd.read_csv(path + 'train_csv' index_col=0 )
test_csv 
submission_csv

x = train_csv.drop(['123'], axis=1)
y = train_csv['123']

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=123,
                                                    shuffle=True
                                                    )

# 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 3))
model.add(Dense(10))
model.add(Dense(1))

# 컴파일, 훈련
loss = model.compile(loss='mse',
                     optimizer='adam',
                     metrics=['acc']
                     )
es = EarlyStopping(monitor='acc',
                   mode='min',
                   patience=10,
                   restore_best_weights=True,
                   verbose=1
                   )
model.fit(x,
          y,
          epochs=100,
          verbose=1,
          validation_batch_size=0.2,
          batch_size=100,
          callbacks=[es]
          )

# 평가, 예측
results = model.predict(x_test, y_test)
submit = model.evaluate(test_csv)
submission_csv['y'] = submit

submitssion_csv.to_csv =(path + 'submission.csv' index = false)











