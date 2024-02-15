# xgboost와 그리드서치, 랜덤서치 ,하빙등을 사용
# m31_2번보다 성능을 좋게 만든다.

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.datasets import mnist
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import time
from xgboost import XGBClassifier
import numpy as np
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Parameters: {min_samples_leaf} are not used.")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 사용할 GPU 번호 설정

(x_train, y_train), (x_test, y_test) = mnist.load_data() # y값 필요 없어서 x값 2개만 받을거다 // _
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

n_splits = 3
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
n_components = 9
lda = LinearDiscriminantAnalysis(n_components=n_components) # n_classes(라벨) 의 갯수에 영향을 받는다. // 차원이 줄어든다. // 분류 부분만 쓴다.
# n_components는 n_features 또는 n_classes - 1 값보다는 커야한다.
lda.fit(x_train, y_train)  # x_train, y_train으로 수정
x_train_lda = lda.transform(x_train)
x_test_lda = lda.transform(x_test)

# 모델 구성
parameters = {
    "n_estimators": [100, 150],           # 트리 개수 증가
    "max_depth": [6, 8],                  # 최대 깊이 증가
    "learning_rate": [0.1, 0.2],          # 학습률 증가
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
    "min_child_weight": [1, 2],           # 최소 자식 노드 가중치 조정
    "gamma": [0.1, 0.2],                  # 감마 값 변경
}
     
model = RandomizedSearchCV(XGBClassifier(tree_method='gpu_hist', device='gpu'), 
                     parameters,
                     cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1,
                     random_state=123,
                     n_iter=5,
                     )
start_time = time.time()
model.fit(x_train_lda, y_train)  # x_train_lda, y_train으로 수정
end_time = time.time()

from sklearn.metrics import accuracy_score
best_predict = model.best_estimator_.predict(x_test_lda)  # x_test_lda로 수정
best_acc_score = accuracy_score(y_test, best_predict)
print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print('best_score :', model.best_score_)
print('score :', model.score(x_test_lda, y_test))  # x_test_lda로 수정

y_predict = model.predict(x_test_lda)  # x_test_lda로 수정
print("accuracy_score :", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test_lda)  # x_test_lda로 수정
print("최적튠 ACC :", accuracy_score(y_test, y_pred_best))
print("걸린시간 :", round(end_time - start_time, 2), "초")

'''
최적의 파라미터 :  {'subsample': 0.8, 'n_estimators': 150, 'min_child_weight': 1, 'max_depth': 8, 'learning_rate': 0.1, 'gamma': 0.2, 'colsample_bytree': 0.8}
best_score : 0.8714666666666666
score : 0.8719
accuracy_score : 0.8719
최적튠 ACC : 0.8719
걸린시간 : 88.17 초

최적의 파라미터 :  {'subsample': 0.8, 'n_estimators': 150, 'min_child_weight': 1, 'max_depth': 8, 'learning_rate': 0.2, 'gamma': 0.1, 'colsample_bytree': 0.9}
best_score : 0.9197833333333335
score : 0.92
accuracy_score : 0.92
최적튠 ACC : 0.92
걸린시간 : 85.71 초
'''


# best_score : 0.9679166666666668
# score : 0.9699
# accuracy_score : 0.9699
# 최적튠 ACC : 0.9699
# 걸린시간 : 104.12 초

# 최적의 파라미터 :  {'subsample': 0.9, 'n_estimators': 100, 'min_child_weight': 1, 'max_depth': 6, 'learning_rate': 0.1, 'gamma': 0.1, 'colsample_bytree': 0.9}
# best_score : 0.9685499999999999
# score : 0.9711
# accuracy_score : 0.9711
# 최적튠 ACC : 0.9711
# 걸린시간 : 217.38 초

# 최적의 파라미터 :  {'tree_method': 'gpu_hist', 'subsample': 0.8, 'n_estimators': 150, 'min_child_weight': 1, 'max_depth': 8, 'learning_rate': 0.2, 'gamma': 0.1, 'device': 'gpu', 'colsample_bytree': 0.9}
# best_score : 0.9720833333333333
# score : 0.9784
# accuracy_score : 0.9784
# 최적튠 ACC : 0.9784
# 걸린시간 : 197.7 초