# 주성분 분석 (Principal Component Analysis), PCA
# 복잡한 데이터를 차원 축소 알고리즘으로 조금 더 심플한 차원의 데이터로 만들어 분석 // 차원 축소
# 데이터의 차원이 증가할 수록 데이터 공간의 부피가 기하 급수적으로 증가하기 때문에, 데이터의 밀도는 차원이 증가할 수록 희소-> 오버피팅


# 스케일링 후 PCA후 스플릿
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

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=2) # n_components = 데이터의 열(특성)의 수 // pca하기전에 스케일링 해야한다. // 통상적으로 스텐다드 // 자기 차원 이상으로 하면 에러
x = pca.fit_transform(x) 
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123,shuffle=True, stratify=y)

# 2. 모델
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train,y_train)

# # 4. 평가 예측
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

results = model.score(x_test, y_test)
print('====================================')
print(x.shape)
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
