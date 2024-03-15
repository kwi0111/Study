import numpy as np
import random
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
import datetime
from skopt import BayesSearchCV

#1
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

path = 'C:\\_data\\dacon\\income\\'

train = pd.read_csv('C:\\_data\\dacon\\income\\train.csv')
test = pd.read_csv('C:\\_data\\dacon\\income\\test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')

labels = train.columns.tolist()
# print(labels)
# ['ID', 'Age', 'Gender', 'Education_Status', 'Employment_Status', 
# 'Working_Week (Yearly)', 'Industry_Status', 'Occupation_Status', 'Race', 'Hispanic_Origin', 
# 'Martial_Status', 'Household_Status', 'Household_Summary', 'Citizenship', 'Birth_Country', 
# 'Birth_Country (Father)', 'Birth_Country (Mother)', 'Tax_Status', 'Gains', 'Losses', 
# 'Dividends', 'Income_Status', 'Income']

# print(pd.value_counts(train['Tax_Status']))
# print(np.unique(train['Race']))

x = train.drop(columns=['ID', 'Income'])
y = train['Income']
test_x = test.drop(columns=['ID'])

encoding_target = list(x.dtypes[x.dtypes == "object"].index)

for i in encoding_target:
    le = LabelEncoder()
    
    # train과 test 데이터셋에서 해당 열의 모든 값을 문자열로 변환
    x[i] = x[i].astype(str)
    test_x[i] = test_x[i].astype(str)
    
    le.fit(x[i])
    x[i] = le.transform(x[i])
    
    # test 데이터의 새로운 카테고리에 대해 le.classes_ 배열에 추가
    for case in np.unique(test_x[i]):
        if case not in le.classes_: 
            le.classes_ = np.append(le.classes_, case)
    
    test_x[i] = le.transform(test_x[i])

# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# test_x = scaler.transform(test_x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 42)

n_splits = 10
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 42 )

# 이건 엑쥐비
param_space = {
    "n_estimators": (100, 1000),          # 트리의 개수 범위를 100에서 1000 사이로 변경
    "max_depth": (5, 50),               # 트리의 최대 깊이 범위를 5에서 50 사이로 변경
    "min_samples_split": (2, 20),       # 노드를 분할하기 위한 최소 샘플 수 범위를 2에서 20 사이로 변경
    "min_samples_leaf": (1, 10),         # 리프 노드가 가져야 하는 최소 샘플 수 범위를 1에서 10 사이로 변경
    "max_features": ['auto', 'sqrt', 'log2'],  # None 제외
    "max_leaf_nodes": (10, 100),         # 리프 노드의 최대 수 범위를 10에서 100 사이로 변경
    "bootstrap": [True],
    "min_impurity_decrease": (0, 0.1),   # 불순도 감소 범위를 0에서 0.1로 변경
    "min_weight_fraction_leaf": (0, 0.5), # 최소 가중치 비율 범위를 0에서 0.5로 변경
    "criterion": ['gini'],
    "learning_rate": (0.01, 0.2),        # 학습률 범위를 0.01에서 0.2로 변경
    "subsample": (0.5, 1.0),             # 트리를 학습하는데 사용할 샘플의 비율 범위를 0.5에서 1.0으로 변경
    "colsample_bytree": (0.5, 1.0),      # 각 트리마다 사용할 피처의 비율 범위를 0.5에서 1.0으로 변경
    "gamma": (0, 10),                    # 트리 노드를 분할하기 위한 최소 손실 감소 값 범위를 0에서 10으로 변경
    "reg_alpha": (0, 1),                 # L1 정규화 항 범위를 0에서 1로 변경
    "reg_lambda": (0, 1),                # L2 정규화 항 범위를 0에서 1로 변경
}

# 이건 캣부스트
# param_space = {
#     "n_estimators": (100, 500),          # 트리의 개수 범위를 100에서 500 사이로 변경
#     "max_depth": (5, 15),               # 트리의 최대 깊이 범위를 5에서 20 사이로 변경
#     "min_child_samples": (2, 10),       # 리프 노드에 필요한 최소 샘플 수 범위를 2에서 10 사이로 변경
#     "max_bin": (5, 20),                 # 각 피처의 이산화 범위를 5에서 20 사이로 변경
#     "grow_policy": ['SymmetricTree', 'Depthwise'],  # 트리 성장 방식 변경
#     "l2_leaf_reg": (1e-6, 100),         # L2 정규화 범위를 1e-6에서 100으로 변경
#     "subsample": (0.6, 0.9),             # 트리를 학습하는데 사용할 샘플의 비율 범위를 0.6에서 0.9으로 변경
#     "colsample_bylevel": (0.6, 0.9),    # 각 트리의 수준(깊이)에서 사용할 피처의 비율 범위를 0.6에서 0.9으로 변경
#     "learning_rate": (0.01, 0.1),        # 학습률 범위를 0.01에서 0.1로 변경
# }





xgb_model = XGBRegressor()
# catboost_model = CatBoostRegressor()

model = BayesSearchCV(
    estimator=xgb_model,
    # estimator=catboost_model,
    search_spaces=param_space,
    n_iter=100,  # 탐색 횟수를 50으로 줄임
    cv=kfold,
    n_jobs=-1,
    verbose=1,
    refit=True
)
#3
model.fit(x_train, y_train) 

#4
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
best_y_predict = model.best_estimator_.predict(x_test)
# mse
mse = mean_squared_error(y_test, y_predict)
best_mse = mean_squared_error(y_test, best_y_predict)
# rmse
rmse = np.sqrt(mse)
best_rmse = np.sqrt(best_mse)
# submission
preds = model.predict(test_x)
best_pred = model.best_estimator_.predict(test_x)

# submission = pd.read_csv('C:\\_data\\dacon\\income\\sample_submission.csv')
submission = submission.head(len(preds))
submission['Income'] = preds
# print(submission)

# submission.to_csv('C:\\_data\\dacon\\income\\submission_0314_1.csv', index=False)
dt = datetime.datetime.now()
submission.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_rmse_{best_rmse:4}.csv",index=False)


print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_) 
print('best_rmse : ', best_rmse)