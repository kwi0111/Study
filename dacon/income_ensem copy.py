
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
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
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




X = x.fillna(0)
test_csv = test_csv.fillna(0)

X = pd.get_dummies(X, drop_first=True)
test = pd.get_dummies(test_csv, drop_first=True)

df = X 
df.columns = df.columns.to_series().apply(lambda x: x.replace('[', '').replace(']', '').replace('<', '')).astype(str)

X, test = X.align(test, join='left', axis=1, fill_value=0)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# 학습 데이터와 테스트 데이터의 열을 맞추기
missing_cols = set(X.columns) - set(test.columns)
for c in missing_cols:
    test[c] = 0
test = test[X.columns]


from lightgbm import LGBMRegressor

class EnsembleRegressor:
    def __init__(self, xgb_params, catboost_params, lgbm_params, xgb_weight, catboost_weight, lgbm_weight):
        self.xgb_model = XGBRegressor(**xgb_params, random_state=42)
        # `catboost_params`에서 'weight' 키가 있다면 제거
        catboost_params_cleaned = {k: v for k, v in catboost_params.items() if k != 'weight'}
        self.catboost_model = CatBoostRegressor(**catboost_params_cleaned, random_state=42, verbose=0)
        self.lgbm_model = LGBMRegressor(**lgbm_params, random_state=42)
        self.xgb_weight = xgb_weight
        self.catboost_weight = catboost_weight
        self.lgbm_weight = lgbm_weight

    def fit(self, X, y):
        self.xgb_model.fit(X, y)
        self.catboost_model.fit(X, y)
        self.lgbm_model.fit(X, y)

    def predict(self, X):
        xgb_pred = self.xgb_model.predict(X)
        catboost_pred = self.catboost_model.predict(X)
        lgbm_pred = self.lgbm_model.predict(X)
        # 가중 평균 계산
        return (self.xgb_weight * xgb_pred + self.catboost_weight * catboost_pred + self.lgbm_weight * lgbm_pred) / \
               (self.xgb_weight + self.catboost_weight + self.lgbm_weight)

from sklearn.model_selection import KFold, ShuffleSplit 
kf = KFold(n_splits=3, shuffle=True, random_state=42)

def objective(trial):
    try:
        # XGBoost 파라미터
        xgb_params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),  # 줄임
            'max_depth': trial.suggest_int('xgb_max_depth', 4, 7),  # 줄임
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.02, 0.05),  # 늘림
            'min_child_weight': trial.suggest_int('xgb_min_child_weight', 2, 6),
            'gamma': trial.suggest_float('xgb_gamma', 0.2, 0.4),
            'subsample': trial.suggest_float('xgb_subsample', 0.75, 0.85),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.75, 0.85),
            'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.001, 1.0),
            'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.001, 1.0),
        }

        # CatBoost 파라미터
        catboost_params = {
            'iterations': trial.suggest_int('catboost_iterations', 100, 500),
            'depth': trial.suggest_int('catboost_depth', 4, 6),
            'learning_rate': trial.suggest_float('catboost_learning_rate', 0.005, 0.05),
            'l2_leaf_reg': trial.suggest_float('catboost_l2_leaf_reg', 3, 5, log=True),
            'border_count': trial.suggest_int('catboost_border_count', 150, 200),
            'bagging_temperature': trial.suggest_float('catboost_bagging_temperature', 0.1, 0.9),
            'random_strength': trial.suggest_float('catboost_random_strength', 1e-9, 1e-3),
        }

        # LightGBM 파라미터
        lgbm_params = {
            'n_estimators': trial.suggest_int('lgbm_n_estimators', 150, 250),
            'max_depth': trial.suggest_int('lgbm_max_depth', 3, 5),
            'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.015, 0.025),
            'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 40),
            'min_child_samples': trial.suggest_int('lgbm_min_child_samples', 20, 60),
            'subsample': trial.suggest_float('lgbm_subsample', 0.75, 0.85),
        }

        
        xgb_weight = trial.suggest_float('xgb_weight', 0, 1)
        catboost_weight = trial.suggest_float('catboost_weight', 0, 1 - xgb_weight)
        lgbm_weight = 1 - xgb_weight - catboost_weight

        model = EnsembleRegressor(
            xgb_params=xgb_params, 
            catboost_params=catboost_params, 
            lgbm_params=lgbm_params,
            xgb_weight=xgb_weight, 
            catboost_weight=catboost_weight, 
            lgbm_weight=lgbm_weight
        )
        
        scores = []
        for train_index, test_index in kf.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[test_index]
            y_train, y_val = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            score = np.sqrt(mean_squared_error(y_val, preds))
            scores.append(score)
        
        # scores 리스트가 비어있지 않은 경우에만 평균을 계산합니다.
        if scores:
            average_score = np.mean(scores)
            return average_score
        else:
            # scores 리스트가 비어있다면 최악의 경우를 가정합니다.
            return float('inf')
    except Exception as e:
        print(f"An error occurred during the trial: {e}")
        # 예외 발생 시 최악의 경우를 가정합니다.
        return float('inf')



study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30000 ,timeout=40000)  # 100회 시도하거나, 총 600초가 경과하면 종료

if len(study.trials) > 0:
    # 최적화된 가중치를 사용하여 모델을 다시 훈련시키고, 테스트 데이터에 대한 예측을 수행합니다.
    best_xgb_params = {k.replace('xgb_', ''): v for k, v in study.best_trial.params.items() if k.startswith('xgb_')}
    best_catboost_params = {k.replace('catboost_', ''): v for k, v in study.best_trial.params.items() if k.startswith('catboost_')}
    best_lgbm_params = {k.replace('lgbm_', ''): v for k, v in study.best_trial.params.items() if k.startswith('lgbm_')}

    best_xgb_weight = study.best_trial.params.get('xgb_weight', 0.33)  # 기본값 설정
    best_catboost_weight = study.best_trial.params.get('catboost_weight', 0.33)
    best_lgbm_weight = 1 - best_xgb_weight - best_catboost_weight

    # EnsembleRegressor 인스턴스 생성
    best_model = EnsembleRegressor(
        xgb_params=best_xgb_params, 
        catboost_params=best_catboost_params, 
        lgbm_params=best_lgbm_params,
        xgb_weight=best_xgb_weight, 
        catboost_weight=best_catboost_weight, 
        lgbm_weight=best_lgbm_weight
    )

    best_model.fit(x_train, y_train)  # 전체 훈련 데이터셋을 사용하여 최종 모델을 훈련합니다.
else:
    print("No trials completed. Adjust 'timeout' or 'n_trials'.")
    
# 최적화된 모델로 테스트 데이터에 대한 예측을 수행합니다.
ensemble_pred = best_model.predict(x_test)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
print(f'Optimized Ensemble RMSE: {ensemble_rmse}')

# 테스트 데이터셋에 대한 예측을 수행합니다.
test_preds = best_model.predict(test)
submission_csv['Income'] = test_preds

# 결과를 CSV 파일로 저장합니다.
dt = datetime.datetime.now()
submission_csv.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_rmse_{ensemble_rmse:4}.csv",index=False)


