import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
path = "C:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv)

print(train_csv.shape) #(5497, 13)
print(test_csv.shape) #(1000, 12)
print(submission_csv.shape) #(1000, 2)


print(train_csv.columns) #'quality', 'fixed acidity', 'volatile acidity', 'citric acid',
    #    'residual sugar', 'chlorides', 'free sulfur dioxide',
    #    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
    #    'type'],
    
x = train_csv.drop(['quality'], axis= 1)
y = train_csv['quality']
y = y - 3

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0

test_csv.loc[test_csv['type'] == 'red', 'type'] = 1
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

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
          eval_metric='mlogloss',
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
        eval_metric = 'mlogloss',
        
    )
    select_model.fit(select_x_train, y_train,
                     eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
                     verbose=0,
                     )
    select_y_predict = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_predict)
    print("Trech=%.3f, n=%d, ACC: %.2f%%" %(i, select_x_train.shape[1], score*100))

'''
Trech=0.055, n=12, ACC: 58.09%
Trech=0.064, n=11, ACC: 59.27%
Trech=0.067, n=10, ACC: 60.09%
Trech=0.069, n=9, ACC: 59.27%
Trech=0.070, n=8, ACC: 58.82%
Trech=0.074, n=7, ACC: 58.36%
Trech=0.074, n=6, ACC: 58.00%
Trech=0.077, n=5, ACC: 57.27%
Trech=0.092, n=4, ACC: 57.09%
Trech=0.097, n=3, ACC: 56.18%
Trech=0.114, n=2, ACC: 53.82%
Trech=0.149, n=1, ACC: 54.27%
'''
