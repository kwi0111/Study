from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn. model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#1.데이터
x, y = load_diabetes(return_X_y=True) 
# x, y = load_breast_cancer(return_X_y=True) 


x_train, x_test, y_train, y_test = train_test_split(x, y , random_state=123, train_size=0.85,
                                                    # stratify=y,
                                                    )

# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

'''
'n_estimators' : [100,200,300,400,500,1000] 디폴트 100 / 1~inf / 정수 // 경사 하강법을 이용하여 트리의 가중치 업데이트 // 에포
'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] 디폴트 0.3 / 0~1 / eta // 통상적으로 작으면 작을수록 좋다. // 가중치를 업데이트하는 속도를 제어
'max_depth' : [None, 2, 3, 4, 5, 6, 7,8, 9,10] 디폴트6 / 0~inf / 정수 // 트리 계열에서 깊이
'gamma' : [0,1,2,3,4,5,6,10,100] 디폴트0 / 0~inf 
'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100] 디폴트 1 / 0~inf
'subsample' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1 ] 디폴트 1 / 0~1
'colsample_bytree' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1 ] 디폴트 1 / 0~1
'colsample_bylevel' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1 ] 디폴트 1 / 0~1
'colsample_bynode' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1 ] 디폴트 1 / 0~1
'reg_alpha' : [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 0 / 0~inf / L1 절대값 가중치 규제 / alpha // 가중치 제한
'reg_lambda' : [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 1 / 0~inf / L2 제곱 가중치 규제 / lambda 
'''

parameters = {
    'n_estimators': 10000,
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight' : 10,
}

#2.모델
# model = XGBClassifier()  # logloss 디폴트
model = XGBRegressor()
model.set_params(
    early_stopping_rounds=10,
    **parameters,
                 ) 

#3. 훈련
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=15,
        #   eval_metric = 'rmse',   # 회귀 디폴트
          eval_metric = 'mae',    # 회귀 // rmsle, mape, mphe 등등..
        
        #   eval_metric = 'mlogloss', # 다중 분류 디폴트 
        #   eval_metric = 'merror',   # 다중분류
        #   eval_metric = 'auc',      # 이진 분류, 다중 분류 // 통상적 이진 분류
        #   eval_metric = 'logloss',  # 이진분류 디폴트 // ACC
        #   eval_metric = 'error',    # 이진분류 : 잘못 분류된 샘플의 비율 // ACC
          )

#4. 평가, 예측
# print("사용 파라미터 : ", model.get_params())
results = model.score(x_test, y_test)
y_pred = model.predict(x_test)

print("최고의 점수1 : ", results)  
from sklearn.metrics import r2_score, accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error

print("r2 : ", r2_score(y_test, y_pred))  
print("mae : ", mean_absolute_error(y_test, y_pred))  
# ValueError: Must have at least 1 validation dataset for early stopping. // 발리데이션 데이터 필요
# 최고의 점수1 :  0.5530691730617214

print("==================================")
hist = model.evals_result()
print(hist)

train_error = hist['validation_0']['mae'] # validation_0은 학습 세트 // train_error 리스트는 학습 중에 각 에폭에서 훈련 데이터에 대한 평가 지표
val_error = hist['validation_1']['mae'] # validation_1은 검증 세트
epochs = range(1, len(train_error) + 1)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_error, label='Training MAE')
plt.plot(epochs, val_error, label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Number of epochs')
plt.ylabel('MAE')
plt.legend()
plt.grid()
plt.show()

from xgboost import plot_importance
plot_importance(model)
plt.show()





