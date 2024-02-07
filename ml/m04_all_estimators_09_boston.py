# 09_1에서 가져옴

# 보스턴에 관한 데이터
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import time
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVC, SVC, LinearSVR
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1. 데이터
datasets = load_boston()        # 변수에 집어 넣은다음 프린트
print(datasets)
x = datasets.data           # x에서 스케일링
y = datasets.target         # y 건들지 않는다.

x_train, x_test, y_train, y_test = train_test_split(x, y,               
                                                    train_size=0.7,
                                                    random_state=1140,     
                                                    shuffle=True,
                                                    )


#2. 모델 구성 
# model = LinearSVR(C=100)
# model = LinearRegression()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()



#3. 컴파일, 훈련 
start_time = time.time()   #현재 시간
model.fit(x_train, y_train)
end_time = time.time()   #끝나는 시간

#4. 평가, 예측
results = model.score(x_test, y_test) 
y_predict = model.predict(x_test) 
print("acc : ", results)
print("걸린시간 : ", round(end_time - start_time, 2),"초")

# acc :  0.5275130859946872
# 걸린시간 :  0.01 초

# 스케일러 x
# acc :  0.19233716312185622
# 걸린시간 :  0.01 초

# LinearSVR                   0.5198578310569075  
# LinearRegression            0.739750714020937
# KNeighborsRegressor         0.5389429273211952
# DecisionTreeRegressor       0.8418772268815504
# RandomForestRegressor        0.9210620914378058