
# 스케일링 후 LDA후 교육용 분류 파일들 만들기

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
import numpy as np
print(sk.__version__)

#1. 데이터
# datasets = load_iris()
# datasets = load_breast_cancer()
datasets = load_digits()    # 컬럼 64개 라벨 10개


x=datasets['data']
y=datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

scaler = StandardScaler()
x = scaler.fit_transform(x)

lda = LinearDiscriminantAnalysis(n_components=9) #  n_classes(라벨) 의 갯수에 영향을 받는다. 4와 3-1 중에 작은거쓴다. // 차원이 줄어든다. // 분류 부분만 쓴다.
# n_components는  min(n_features, n_classes - 1)보다 클수 없다. // 입력 데이터와 해당하는 레이블(클래스)이 필요
lda.fit(x,y)
x = lda.transform(x)
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

evr = lda.explained_variance_ratio_
print(evr)
print(sum(evr))

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

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


