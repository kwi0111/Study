# 주성분 분석 (Principal Component Analysis), PCA
# 복잡한 데이터를 차원 축소 알고리즘으로 조금 더 심플한 차원의 데이터로 만들어 분석 // 차원 축소

# 스케일링 후 PCA후 스플릿
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
print(sk.__version__)

#1. 데이터
# datasets = load_diabetes()
datasets = load_breast_cancer()


x=datasets['data']
y=datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=30) #  pca하기전에 스케일링 해야한다. // 통상적으로 스텐다드 // 자기 차원 이상으로 하면 에러
x = pca.fit_transform(x) 
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123,shuffle=True)

# 2. 모델
# model = RandomForestRegressor()
model = RandomForestClassifier()


# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가 예측
results = model.score(x_test, y_test)
print('====================================')
print(x.shape)
print('model.score : ', results)

evr = pca.explained_variance_ratio_
# PCA(주성분 분석)를 통해 변환된 각 주성분이 전체 데이터의 분산을 설명하는 비율
print(evr)
print(sum(evr))

import matplotlib.pyplot as plt
evr_cumsum = np.cumsum(evr)
print(evr_cumsum)
plt.plot(evr_cumsum)
plt.grid()
plt.show()

# [0.40242108 0.14923197 0.12059663 0.09554764 0.06621814 0.06027171
#  0.05365657 0.0433682  0.007832   0.00085607]

# for i in range(x.shape[1], 0, -1):
#     pca = PCA(n_components=i)
#     x_train = pca.fit_transform(x_train)
#     x_test = pca.transform(x_test)
    
#     # 모델 초기화
#     model = RandomForestClassifier()
    
#     # 모델 훈련
#     model.fit(x_train, y_train)
    
#     # 모델 평가
#     results = model.score(x_test, y_test)
#     print('====================================')
#     print(x_train.shape)
#     print('model.score : ', results)


# ====================================
# (442, 10) pca 안했따
# model.score :  0.511942510804726
# ====================================
# (442, 4)
# model.score :  0.46992522100698675
# ====================================
# (442, 8)
# model.score :  0.5251458328759518



# ====================================
# (150, 4)
# model.score :  0.9333333333333333
# ====================================
# (150, 3)
# model.score :  0.8666666666666667
# ====================================
# (150, 2)
# model.score :  0.8
# ====================================
# (150, 1)
# model.score :  0.8666666666666667
