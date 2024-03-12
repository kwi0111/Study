# 보스턴에 관한 데이터
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_boston, load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()        # 변수에 집어 넣은다음 프린트
print(datasets)
x = datasets.data           # x에서 스케일링
y = datasets.target         # y 건들지 않는다.
print(x)
print(x.shape)  #(506, 13) 컬럼이 무엇인지 모름.
print(y)
print(x.shape, y.shape)  #(506, 13), (506,)
print(datasets.feature_names)
print(datasets.DESCR)   #설명하다. 묘사하다. 데이터셋의 내용

x_train, x_test, y_train, y_test = train_test_split(x, y,               
                                                    train_size=0.7,
                                                    random_state=1140,     
                                                    shuffle=True,
                                                    )
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
scaler = MinMaxScaler() # 클래스 정의
# scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델 구성 
model = Sequential()
model.add(Dense(5, input_dim=30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation= 'sigmoid'))

# 3. 컴파일, 훈련 
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=20,
                   verbose=1,
                   restore_best_weights=True
                   )
mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath='../_data/_save/MCP/keras26_boston_MCP1.hdf5',
                      period=20,
                      )


rlr = ReduceLROnPlateau(monitor='val_loss',
                        patience=10,
                        mode='auto',
                        verbose=1,
                        factor=0.5,  # 반으로 줄여라.
                        
                        )

from keras.optimizers import Adam
lr = 0.01

model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=lr))
model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.18, callbacks=[es, mcp, rlr])
model = load_model('../_data/_save/MCP/keras26_boston_MCP1.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
results = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("로스 : ", loss)
print("r2 스코어 : " , r2)

# 로스 :  0.44418686628341675
# r2 스코어 :  0.43411901282590437



