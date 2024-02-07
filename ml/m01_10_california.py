import numpy as np                                                  # numpy 빠른 계산을 위해 지원되는 파이썬 라이브러리
import time
from keras.models import Sequential
from keras.layers import Dense

from sklearn.datasets import fetch_california_housing               # 사이킷런 : 파이썬 머신러닝 라이브러리 // sklearn에서 제공하는 데이터셋
from sklearn.model_selection import train_test_split                # scikit-learn 패키지 중 model_selection에서 데이터 분할
from sklearn.svm import LinearSVR



#1. 데이터
datasets = fetch_california_housing()                               # fetch : 가져옴
x = datasets.data                                                   # 샘플 데이터
y = datasets.target                                                 # 라벨 데이터


x_train, x_test, y_train, y_test = train_test_split(x, y,                     # 훈련 데이터, 테스트 데이터 나누는 과정
                                                    train_size=0.7,
                                                    random_state=123,     
                                                    shuffle=True,
                                                    )
# from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
# from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# scaler = MinMaxScaler() # 클래스 정의
# # scaler = StandardScaler() # 클래스 정의
# # scaler = MaxAbsScaler() # 클래스 정의
# # scaler = RobustScaler() # 클래스 정의

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


#2. 모델 구성 
model = LinearSVR(C=100)


#3. 컴파일, 훈련 
start_time = time.time()   #현재 시간
model.fit(x_train, y_train)
end_time = time.time()   #끝나는 시간                                                          # 끝나는 시간

#4. 평가, 예측
results = model.score(x_test, y_test) 
y_predict = model.predict(x_test) 
print("acc : ", results)
print("걸린시간 : ", round(end_time - start_time, 2),"초")    
from sklearn.metrics import r2_score  
r2 = r2_score(y_test, y_predict)                                                # 실제값, 예측값 순서
print("r2 스코어 : " , r2)

# 민맥스
# acc :  0.5608205401471694
# 걸린시간 :  0.26 초
# r2 스코어 :  0.5608205401471694

# 스케일러 x
# acc :  -0.16543373046825738
# 걸린시간 :  0.36 초
# r2 스코어 :  -0.16543373046825738