
import optuna
import numpy as np
import random
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from skopt import BayesSearchCV
import datetime
from sklearn.preprocessing import StandardScaler


# 데이터 로드
path = 'C:\\_data\\dacon\\income\\'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

test_csv = test_csv.fillna(method= 'bfill')


train_csv['Citizenship'] = train_csv['Citizenship'].apply(lambda x: 'Native' if 'Native' in x else x)
test_csv['Citizenship'] = test_csv['Citizenship'].apply(lambda x: 'Native' if 'Native' in x else x)


label_encoder_dict = {}
for label in train_csv:
    data = train_csv[label].copy()
    if data.dtypes == 'object':
        label_encoder = LabelEncoder()
        train_csv[label] = label_encoder.fit_transform(data)
        label_encoder_dict[label] = label_encoder
        
for label, encoder in label_encoder_dict.items():
    data = test_csv[label]
    test_csv[label] = encoder.transform(data)

x = train_csv.drop(['Household_Summary','Income'], axis=1)
y = train_csv['Income']
test_csv = test_csv.drop(['Household_Summary'], axis=1)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

class EnsembleRegressor:
    def __init__(self, xgb_params, catboost_params, xgb_weight, catboost_weight):
        self.xgb_model = XGBRegressor(**xgb_params, random_state=42, enable_categorical=True)
        self.catboost_model = CatBoostRegressor(**catboost_params, random_state=42, verbose=1)
        self.xgb_weight = xgb_weight
        self.catboost_weight = catboost_weight

    def fit(self, X, y):
        self.xgb_model.fit(X, y)
        self.catboost_model.fit(X, y)

    def predict(self, X):
        xgb_pred = self.xgb_model.predict(X)
        catboost_pred = self.catboost_model.predict(X)
        # 가중 평균 계산
        return (self.xgb_weight * xgb_pred + self.catboost_weight * catboost_pred) / (self.xgb_weight + self.catboost_weight)


def objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 500, 1500),  # 더 넓은 범위로 수정
        'max_depth': trial.suggest_int('xgb_max_depth', 4, 12),  # 수정됨
        'learning_rate': trial.suggest_loguniform('xgb_learning_rate', 0.005, 0.05),  # 좁은 범위로 수정
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 2, 10),  # 범위 수정
        'gamma': trial.suggest_float('xgb_gamma', 0.1, 0.5),  # 조금 더 넓은 범위로 수정
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 0.9),  # 범위 수정
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 0.9),  # 범위 수정
        'reg_alpha': trial.suggest_loguniform('xgb_reg_alpha', 1e-3, 1),  # 수정됨
        'reg_lambda': trial.suggest_loguniform('xgb_reg_lambda', 1, 10),  # 수정됨
    }
    
    catboost_params = {
        'iterations': trial.suggest_int('catboost_iterations', 500, 1500),  # 더 넓은 범위로 수정
        'depth': trial.suggest_int('catboost_depth', 4, 10),  # 유지
        'learning_rate': trial.suggest_loguniform('catboost_learning_rate', 0.005, 0.05),  # 수정됨
        'l2_leaf_reg': trial.suggest_loguniform('catboost_l2_leaf_reg', 3, 10),  # 수정됨
        'border_count': trial.suggest_int('catboost_border_count', 100, 255),  # 수정됨
        'bagging_temperature': trial.suggest_float('catboost_bagging_temperature', 0.0, 1.0),  # 유지
        'random_strength': trial.suggest_float('catboost_random_strength', 1e-9, 1e-3),  # 수정됨
        'task_type': 'GPU',  # 유지
        'loss_function': 'RMSE',  # 유지
    }   

    
    xgb_weight = trial.suggest_float('xgb_weight', 0, 1)
    catboost_weight = 1 - xgb_weight  # 두 모델의 가중치 합이 1이 되도록 설정
    model = EnsembleRegressor(xgb_params=xgb_params, catboost_params=catboost_params, xgb_weight=xgb_weight, catboost_weight=catboost_weight)

    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

    # return objective

    # model = EnsembleRegressor(xgb_params=xgb_params, catboost_params=catboost_params, xgb_weight=xgb_weight, catboost_weight=catboost_weight)
    
    # # model = EnsembleRegressor(xgb_params=xgb_params, catboost_params=catboost_params)
    # model.fit(x_train, y_train)
    # preds = model.predict(x_test)
    # rmse = np.sqrt(mean_squared_error(y_test, preds))
    # return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30) #,timeout=600)  # 100회 시도하거나, 총 600초가 경과하면 종료

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# 최적의 하이퍼파라미터를 사용하여 모델을 다시 훈련시키고, 테스트 데이터에 대한 예측을 수행합니다.
best_xgb_params = {k.replace('xgb_', ''): v for k, v in study.best_trial.params.items() if k.startswith('xgb_')}
best_catboost_params = {k.replace('catboost_', ''): v for k, v in study.best_trial.params.items() if k.startswith('catboost_')}
best_xgb_weight = study.best_trial.params.get('xgb_weight')
best_catboost_weight = 1 - best_xgb_weight

best_model = EnsembleRegressor(xgb_params=best_xgb_params, catboost_params=best_catboost_params, xgb_weight=best_xgb_weight, catboost_weight=best_catboost_weight)
best_model.fit(x_train, y_train)  # 전체 데이터셋을 사용하여 최종 모델 훈련

# 최적화된 모델로 테스트 데이터에 대한 예측 수행
ensemble_pred = best_model.predict(x_test)

ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
print(f'Optimized Ensemble RMSE: {ensemble_rmse}')

# 테스트 데이터에 대한 예측 수행
test_preds = best_model.predict(test_csv)
submission_csv['Income'] = test_preds

# 제출 파일 생성
dt = datetime.datetime.now()
submission_csv.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_rmse_{ensemble_rmse:4}.csv",index=False)

