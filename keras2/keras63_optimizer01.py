# 보스턴에 관한 데이터
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_breast_cancer
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_breast_cancer()        # 변수에 집어 넣은다음 프린트
print(datasets)
x = datasets.data           # x에서 스케일링
y = datasets.target         # y 건들지 않는다.

x_train, x_test, y_train, y_test = train_test_split(x, y,               
                                                    train_size=0.7,
                                                    random_state=1140,     
                                                    shuffle=True,
                                                    )
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
# scaler = MinMaxScaler() # 클래스 정의
scaler = StandardScaler() # 클래스 정의
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
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation= 'sigmoid'))

# 3. 컴파일, 훈련 
from keras.optimizers import Adam
learning_rate = 0.0001


model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=learning_rate))
model.fit(x_train, y_train, epochs=200, batch_size=32, validation_split=0.18)




# model = load_model('../_data/_save/MCP/keras26_boston_MCP1.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
results = model.predict(x)

acc = accuracy_score(y_test, y_predict)
print("lr : {0}, 로스 :{1} ".format(learning_rate, loss))
print("lr : {0}, ACC : {1}".format(learning_rate, acc))


'''
lr : 1.0, 로스 :0.6691762208938599
lr : 1.0, ACC : 0.36257309941520466

lr : 0.1, 로스 :0.6554032564163208
lr : 0.1, ACC : 0.36257309941520466

lr : 0.01, 로스 :3.369913101196289 
lr : 0.01, ACC : 0.36257309941520466

lr : 0.001, 로스 :1.4519636631011963
lr : 0.001, ACC : 0.36257309941520466

lr : 0.0001, 로스 :0.4526303708553314
lr : 0.0001, ACC : 0.36257309941520466
'''
