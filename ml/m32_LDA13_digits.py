from sklearn.datasets import load_digits
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time

datasets = load_digits()

x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape) # (1797, 64) (1797,)
print(pd.value_counts(y, sort=False))
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

# ''' 25퍼 미만 열 삭제 '''
# columns = datasets.feature_names
# # columns = x.columns
# x = pd.DataFrame(x,columns=columns)
# print("x.shape",x.shape)
# # ''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
# fi_str = "0.         0.04228381 0.00888499 0.00593978 0.00610986 0.03531101\
#  0.01032542 0.00884368 0.         0.00941933 0.00920098 0.00683095\
#  0.0095273  0.00980344 0.01315444 0.00542787 0.         0.00611506\
#  0.0102995  0.03312828 0.00857941 0.04274852 0.01065067 0.\
#  0.         0.00290121 0.02703704 0.00725884 0.04658638 0.0228351\
#  0.01990054 0.         0.         0.08098435 0.0115676  0.01239669\
#  0.0408804  0.01070477 0.02800508 0.         0.         0.00945863\
#  0.03300425 0.04443023 0.00833293 0.01907562 0.02225556 0.\
#  0.         0.00467057 0.00424119 0.00608522 0.011416   0.02688941\
#  0.03167028 0.         0.         0.00124175 0.03023438 0.00577911\
#  0.05697288 0.00806607 0.03529015 0.03724349"
 
# ''' str에서 숫자로 변환하는 구간 '''
# fi_str = fi_str.split()
# fi_float = [float(s) for s in fi_str]
# print(fi_float)
# fi_list = pd.Series(fi_float)

# ''' 25퍼 미만 인덱스 구하기 '''
# low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
# print('low_idx_list',low_idx_list)

# ''' 25퍼 미만 제거하기 '''
# low_col_list = [x.columns[index] for index in low_idx_list]
# # 이건 혹여 중복되는 값들이 많아 25퍼이상으로 넘어갈시 25퍼로 자르기
# if len(low_col_list) > len(x.columns) * 0.25:   
#     low_col_list = low_col_list[:int(len(x.columns)*0.25)]
# print('low_col_list',low_col_list)
# x.drop(low_col_list,axis=1,inplace=True)
# print("after x.shape",x.shape)

print(np.unique(y, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
n_components = 6
lda = LinearDiscriminantAnalysis(n_components=n_components) #  n_classes(라벨) 의 갯수에 영향을 받는다. // 차원이 줄어든다. // 분류 부분만 쓴다.
# n_components는  n_features 또는 n_classes -1 값보다는 커야한다.
lda.fit(x,y)
x = lda.transform(x)

x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        train_size=0.7,
        random_state=200,    
        stratify=y,
        shuffle=True,
        )


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler


#2. 모델구성
for i in range(x.shape[1], 0, -1):
    # pca = PCA(n_components=i)
    # x_train = pca.fit_transform(x_train)
    # x_test = pca.transform(x_test)
    
    # 모델 초기화
    model = RandomForestClassifier()
    
    # 모델 훈련
    model.fit(x_train, y_train)
    
    # 모델 평가
    results = model.score(x_test, y_test)
    print('====================================')
    print(x_train.shape)
    print('model.score : ', results)
    break

evr = lda.explained_variance_ratio_
print(evr)
print(sum(evr))

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)