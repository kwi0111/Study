import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
path = "C:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


print(train_csv.shape) #(5497, 13)
print(test_csv.shape) #(1000, 12)
print(submission_csv.shape) #(1000, 2)


print(train_csv.columns) #'quality', 'fixed acidity', 'volatile acidity', 'citric acid',
    #    'residual sugar', 'chlorides', 'free sulfur dioxide',
    #    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
    #    'type'],
    
x = train_csv.drop(['quality'], axis= 1)
y = train_csv['quality']

y = LabelEncoder().fit_transform(y)

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

# parameters = {
#     'n_estimators': 1000,
#     'learning_rate': 0.01,
#     'max_depth': 3, # 트리 깊이
#     'gamma' : 0,
#     'min_child_weight' : 10,
#     'min_child_weight' : 0,
#     'subsample' : 0.4,
#     'colsample_bytree' : 0.8,
#     'colsample_bylevel' : 0.7,
#     'colsample_bynode' : 1,
#     'reg_alpha' : 0,
#     'reg_lambda' : 1,
#     'random_state' : 3377,
#     'verbose' : 0,
# }
parameters = {'learning_rate': 0.13349839953884737,
                'n_estimators': 99,
                'max_depth': 8,
                'min_child_weight': 3.471164143831403e-06,
                'subsample': 0.6661302167437514,            #dropout 비슷
                'colsample_bytree': 0.9856906281904222,
                'gamma': 4.5485144879936555e-06,
                'reg_alpha': 0.014276113125688179,
                'reg_lambda': 10.121476098960851,
                # 'nthread' : 20,
                'tree_method' : 'gpu_hist',
                'predictor' : 'gpu_predictor',
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
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score : ', acc)

##############################################
# print(model.feature_importances_)

# thresholds = np.sort(model.feature_importances_)    # 내림차순

# from sklearn.feature_selection import SelectFromModel # 크거나 같은값의 피처는 삭제해버린다.
# print("="*100)
# for i in thresholds:
#     selection = SelectFromModel(model, threshold=i, prefit=False)   # 클래스를 인스턴스화 한다 // 
    
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)
#     # print(i, "\t변형된 x_train: ", select_x_train.shape, "변형된 x_test: ", select_x_test.shape )
    
#     select_model =XGBClassifier()
#     select_model.set_params(
#         early_stopping_rounds=10,
#         **parameters,
#         eval_metric = 'mlogloss',
        
#     )
#     select_model.fit(select_x_train, y_train,
#                      eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
#                      verbose=0,
#                      )
#     select_y_predict = select_model.predict(select_x_test)
#     score = accuracy_score(y_test, select_y_predict)
#     print("Trech=%.3f, n=%d, ACC: %.2f%%" %(i, select_x_train.shape[1], score*100))

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




