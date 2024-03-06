import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
x, y = fetch_covtype(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y , random_state=777, train_size=0.8,
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 3, # 트리 깊이
    'gamma' : 0,
    'min_child_weight' : 10,
    'min_child_weight' : 0,
    'subsample' : 0.4,
    'colsample_bytree' : 0.8,
    'colsample_bylevel' : 0.7,
    'colsample_bynode' : 1,
    'reg_alpha' : 0,
    'reg_lambda' : 1,
    'random_state' : 3377,
    'verbose' : 0,
}


#2. 모델
model = XGBClassifier()
model.set_params(early_stopping_rounds=10, **parameters)

#.3 훈련
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=1,
          eval_metric='logloss',
          )

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)
# 최종점수 :  0.9298245614035088
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score : ', acc)

##############################################
print(model.feature_importances_)

thresholds = np.sort(model.feature_importances_)    # 내림차순
# print(thresholds)
# [0.01226016 0.01844304 0.01364688 0.04408791 0.01009422 0.01062047
#  0.03175311 0.06426384 0.00957733 0.01629062 0.01834335 0.01561584
#  0.01365232 0.03140673 0.01297489 0.01083888 0.01846627 0.01291327
#  0.01083225 0.01474823 0.11274869 0.02523725 0.13143815 0.10594388
#  0.01828141 0.02078197 0.04221498 0.11370341 0.02124572 0.0175748 ]
# [0.00957733 0.01009422 0.01062047 0.01083225 0.01083888 0.01226016
#  0.01291327 0.01297489 0.01364688 0.01365232 0.01474823 0.01561584
#  0.01629062 0.0175748  0.01828141 0.01834335 0.01844304 0.01846627
#  0.02078197 0.02124572 0.02523725 0.03140673 0.03175311 0.04221498
#  0.04408791 0.06426384 0.10594388 0.11274869 0.11370341 0.13143815]
from sklearn.feature_selection import SelectFromModel # 크거나 같은값의 피처는 삭제해버린다.
print("="*100)
for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)   # 클래스를 인스턴스화 한다 // 
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(i, "\t변형된 x_train: ", select_x_train.shape, "변형된 x_test: ", select_x_test.shape )
    
    select_model =XGBClassifier()
    select_model.set_params(
        early_stopping_rounds=10,
        **parameters,
        eval_metric = 'logloss',
        
    )
    select_model.fit(select_x_train, y_train,
                     eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
                     verbose=0,
                     )
    select_y_predict = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_predict)
    print("Trech=%.3f, n=%d, ACC: %.2f%%" %(i, select_x_train.shape[1], score*100))

'''
Trech=0.000, n=54, ACC: 74.60%
Trech=0.000, n=54, ACC: 74.60%
Trech=0.000, n=54, ACC: 74.60%
Trech=0.000, n=54, ACC: 74.60%
Trech=0.000, n=50, ACC: 74.57%
Trech=0.000, n=49, ACC: 74.62%
Trech=0.002, n=48, ACC: 74.55%
Trech=0.002, n=47, ACC: 74.54%
Trech=0.002, n=46, ACC: 74.56%
Trech=0.003, n=45, ACC: 74.61%
Trech=0.003, n=44, ACC: 74.64%
Trech=0.003, n=43, ACC: 74.52%
Trech=0.005, n=42, ACC: 74.55%
Trech=0.005, n=41, ACC: 74.60%
Trech=0.005, n=40, ACC: 74.54%
Trech=0.006, n=39, ACC: 74.53%
Trech=0.006, n=38, ACC: 74.50%
Trech=0.006, n=37, ACC: 74.56%
Trech=0.006, n=36, ACC: 74.48%
Trech=0.007, n=35, ACC: 74.57%
Trech=0.008, n=34, ACC: 74.61%
Trech=0.008, n=33, ACC: 74.52%
Trech=0.008, n=32, ACC: 74.51%
Trech=0.008, n=31, ACC: 74.58%
Trech=0.009, n=30, ACC: 74.62%
Trech=0.009, n=29, ACC: 74.60%
Trech=0.009, n=28, ACC: 74.57%
Trech=0.010, n=27, ACC: 74.49%
Trech=0.011, n=26, ACC: 74.47%
Trech=0.011, n=25, ACC: 74.52%
Trech=0.012, n=24, ACC: 74.07%
Trech=0.013, n=23, ACC: 73.98%
Trech=0.014, n=22, ACC: 73.96%
Trech=0.015, n=21, ACC: 73.11%
Trech=0.016, n=20, ACC: 73.02%
Trech=0.017, n=19, ACC: 72.02%
Trech=0.017, n=18, ACC: 71.91%
Trech=0.018, n=17, ACC: 71.85%
Trech=0.018, n=16, ACC: 71.02%
Trech=0.019, n=15, ACC: 71.07%
Trech=0.022, n=14, ACC: 70.84%
Trech=0.024, n=13, ACC: 70.56%
Trech=0.028, n=12, ACC: 70.53%
Trech=0.031, n=11, ACC: 70.55%
Trech=0.033, n=10, ACC: 69.79%
Trech=0.045, n=9, ACC: 68.97%
Trech=0.046, n=8, ACC: 68.64%
Trech=0.049, n=7, ACC: 68.36%
Trech=0.051, n=6, ACC: 68.45%
Trech=0.055, n=5, ACC: 68.48%
Trech=0.061, n=4, ACC: 67.83%
Trech=0.068, n=3, ACC: 67.44%
Trech=0.084, n=2, ACC: 66.98%
Trech=0.097, n=1, ACC: 51.68%
'''
