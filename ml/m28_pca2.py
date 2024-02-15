# 주성분 분석 (Principal Component Analysis), PCA
# 복잡한 데이터를 차원 축소 알고리즘으로 조금 더 심플한 차원의 데이터로 만들어 분석 // 차원 축소

#스플릿 후 스케일링 후 PCA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
print(sk.__version__)

#1. 데이터
datasets = load_iris()

x=datasets['data']
y=datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

# scaler = StandardScaler()
# x = scaler.fit_transform(x)

# pca = PCA(n_components=1) #  pca하기전에 스케일링 해야한다. // 통상적으로 스텐다드 // 자기 차원 이상으로 하면 에러
# x = pca.fit_transform(x) 
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123,shuffle=True, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

pca = PCA(n_components=3) #  pca하기전에 스케일링 해야한다. // 통상적으로 스텐다드 // 자기 차원 이상으로 하면 에러
x_train = pca.fit_transform(x_train) 
x_test = pca.fit_transform(x_test) 
# print(x)

#2. 모델
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가 예측
results = model.score(x_test, y_test)
print('====================================')
print(x_train.shape)
print('model.score : ', results)
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


# ====================================
# (120, 4)
# model.score :  0.9333333333333333
# ====================================
# (120, 3)
# model.score :  0.8666666666666667