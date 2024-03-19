
'''
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

# 초기 설정 함수
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything()  # 시드 고정

#1.데이터
path = 'C:\\_data\\dacon\\income\\'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv')

# 데이터 준비
x = train.drop(columns=['Income'])
y = train['Income']
test_x = test.copy()

# `y`에서 결측치 확인 및 제거
if y.isnull().any():
    print("Removing rows with NaN values in target variable.")
    mask = ~y.isnull()  # `y`에서 결측치가 아닌 값을 찾는 마스크
    x = x[mask]
    y = y[mask]

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

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# 사용자 정의 앙상블 모델 생성 (개선된 버전)
class MyEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **params):
        self.params = params
    
    def fit(self, X, y):
        self.xgb_params = {k[4:]: v for k, v in self.params.items() if k.startswith('xgb_')}
        self.catboost_params = {k[9:]: v for k, v in self.params.items() if k.startswith('catboost_')}
        
        self.xgb_model = XGBRegressor(**self.xgb_params, random_state=42)
        self.xgb_model.fit(X, y)
        
        self.catboost_model = CatBoostRegressor(**self.catboost_params, random_state=42, verbose=0)
        self.catboost_model.fit(X, y)
        
        return self

    def predict(self, X):
        xgb_pred = self.xgb_model.predict(X)
        catboost_pred = self.catboost_model.predict(X)
        return (xgb_pred + catboost_pred) / 2

    # `get_params`와 `set_params` 오버라이딩
    def get_params(self, deep=True):
        return self.params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.params = parameters
        return self

# 하이퍼파라미터 탐색 공간 설정 (개선된 버전)
xgb_search_space = {
    'xgb_n_estimators': (100, 1200),
    'xgb_max_depth': (3, 12),
    'xgb_learning_rate': (0.01, 0.3),
    'xgb_gamma': (0, 1),
    'xgb_subsample': (0.5, 1.0),
    'xgb_colsample_bytree': (0.3, 1.0),
    'xgb_min_child_weight': (1, 10),
    'xgb_colsample_bylevel': (0.3, 1.0),
}

catboost_search_space = {
    'catboost_n_estimators': (100, 1200),
    'catboost_learning_rate': (0.01, 0.3),
    'catboost_depth': (4, 12),
    'catboost_l2_leaf_reg': (1, 20),
    'catboost_border_count': (100, 300),
    'catboost_bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],  # Categorical parameter
    'catboost_grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']  # Categorical parameter
}

# 전체 하이퍼파라미터 탐색 공간 결합
search_spaces = {**xgb_search_space, **catboost_search_space}

# BayesSearchCV를 사용한 하이퍼파라미터 최적화
opt = BayesSearchCV(
    estimator=MyEnsembleRegressor(),
    search_spaces=search_spaces,
    n_iter=64,  # 탐색 횟수 설정
    cv=5,  # 교차 검증 분할 수
    n_jobs=-1,
    verbose=1
)

# 모델 정의
xgb_model = XGBRegressor(random_state=42)
catboost_model = CatBoostRegressor(random_state=42, verbose=1)

# BayesSearchCV를 사용한 하이퍼파라미터 최적화 실행
opt.fit(x_train, y_train)

# 최적화된 하이퍼파라미터를 바탕으로 최적의 모델 획득
best_ensemble_model = opt.best_estimator_

# 최적화된 모델로 테스트 데이터에 대한 예측 수행
ensemble_pred = best_ensemble_model.predict(x_test)

# 평가
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
print(f'Optimized Ensemble RMSE: {ensemble_rmse}')

# 테스트 데이터에 대한 최종 예측
final_pred = best_ensemble_model.predict(test_x)

# 제출 파일 생성
submission['Income'] = final_pred
dt = datetime.datetime.now()
submission.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_rmse_{ensemble_rmse:4}.csv",index=False)

# Optimized Ensemble RMSE: 588.8807633804358
'''

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
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv')


# 입력과 출력 정의
X = train.drop(columns=['Income'])
y = train['Income']

print(X.dtypes)
X = X.fillna(0)
test = test.fillna(0)

X = pd.get_dummies(X, drop_first=True)
test = pd.get_dummies(test, drop_first=True)

df = X 
df.columns = df.columns.to_series().apply(lambda x: x.replace('[', '').replace(']', '').replace('<', '')).astype(str)

X, test = X.align(test, join='left', axis=1, fill_value=0)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 스케일링 (좋은지 안좋은지 아직 모름)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns)



# 학습 데이터와 테스트 데이터의 열을 맞추기
missing_cols = set(X.columns) - set(test.columns)
for c in missing_cols:
    test[c] = 0
test = test[X.columns]


class EnsembleRegressor:
    def __init__(self, xgb_params, catboost_params, xgb_weight, catboost_weight):
        self.xgb_model = XGBRegressor(**xgb_params, random_state=42, enable_categorical=True)
        self.catboost_model = CatBoostRegressor(**catboost_params, random_state=42, verbose=1)
        self.xgb_weight = xgb_weight
        self.catboost_weight = catboost_weight

    def fit(self, X, y, X_val, y_val):
        # XGBoost에 대한 조기 종료 설정
        self.xgb_model.fit(X, y, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
        
        # CatBoost에 대한 조기 종료 설정
        self.catboost_model.fit(X, y, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)


    def predict(self, X):
        xgb_pred = self.xgb_model.predict(X)
        catboost_pred = self.catboost_model.predict(X)
        # 가중 평균 계산
        return (self.xgb_weight * xgb_pred + self.catboost_weight * catboost_pred) / (self.xgb_weight + self.catboost_weight)


def objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1500),  # 범위 확장
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 15),  # 범위 확장
        'learning_rate': trial.suggest_loguniform('xgb_learning_rate', 0.005, 0.3),  # log scale로 변경
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 0, 10),  # 최소값 조정
        'gamma': trial.suggest_float('xgb_gamma', 0, 1.5),  # 범위 확장
        'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),  # 범위 확장
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0),  # 범위 확장
        'reg_alpha': trial.suggest_loguniform('xgb_reg_alpha', 1e-5, 1),  # log scale로 변경 및 범위 확장
        'reg_lambda': trial.suggest_loguniform('xgb_reg_lambda', 0.1, 10),  # log scale로 변경 및 범위 확장
        'max_delta_step': trial.suggest_int('xgb_max_delta_step', 0, 10),  # 범위 확장
        'eval_metric': trial.suggest_categorical('xgb_eval_metric', ['rmse', 'mae', 'logloss']),  # 옵션 추가
    }
    
    catboost_params = {
        'iterations': trial.suggest_int('catboost_iterations', 100, 1500),  # 범위 확장
        'depth': trial.suggest_int('catboost_depth', 4, 12),  # 범위 확장
        'learning_rate': trial.suggest_loguniform('catboost_learning_rate', 0.005, 0.3),  # log scale로 변경
        'l2_leaf_reg': trial.suggest_loguniform('catboost_l2_leaf_reg', 1, 30),  # log scale로 변경 및 범위 확장
        'border_count': trial.suggest_int('catboost_border_count', 32, 255),  # 범위 확장
        'bagging_temperature': trial.suggest_float('catboost_bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_loguniform('catboost_random_strength', 1e-10, 10),  # log scale로 변경
        'one_hot_max_size': trial.suggest_int('catboost_one_hot_max_size', 2, 25),  # 범위 확장
        'task_type': 'GPU',  # GPU 사용 설정
        'loss_function': trial.suggest_categorical('catboost_loss_function', ['RMSE', 'MAE', 'Quantile']),
    }
    
    xgb_weight = trial.suggest_float('xgb_weight', 0, 1)
    catboost_weight = 1 - xgb_weight  # 두 모델의 가중치 합이 1이 되도록 설정
    model = EnsembleRegressor(xgb_params=xgb_params, catboost_params=catboost_params, xgb_weight=xgb_weight, catboost_weight=catboost_weight)

    
    X_train, X_val, y_train1, y_val = train_test_split(x_train_scaled, y_train1, test_size=0.2, random_state=42)
    model.fit(X_train, y_train1, X_val, y_val)
    # 스케일링된 테스트 데이터로 예측
    preds = model.predict(x_test_scaled)
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
study.optimize(objective, n_trials=1 ,timeout=1200)  # 100회 시도하거나, 총 600초가 경과하면 종료

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# 최적의 하이퍼파라미터를 사용하여 모델을 다시 훈련시키고, 테스트 데이터에 대한 예측을 수행합니다.
best_xgb_params = {k.replace('xgb_', ''): v for k, v in study.best_trial.params.items() if k.startswith('xgb_')}
best_catboost_params = {k.replace('catboost_', ''): v for k, v in study.best_trial.params.items() if k.startswith('catboost_')}
best_xgb_weight = study.best_trial.params.get('xgb_weight')
best_catboost_weight = 1 - best_xgb_weight

X_train, X_val, y_train, y_val = train_test_split(x_train_scaled, y_train, test_size=0.2, random_state=42)
best_model = EnsembleRegressor(xgb_params=best_xgb_params, catboost_params=best_catboost_params, xgb_weight=best_xgb_weight, catboost_weight=best_catboost_weight)
best_model.fit(x_train_scaled, y_train)  # 전체 데이터셋을 사용하여 최종 모델 훈련

# 최적화된 모델로 테스트 데이터에 대한 예측 수행
ensemble_pred = best_model.predict(x_test_scaled)

ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
print(f'Optimized Ensemble RMSE: {ensemble_rmse}')

# 테스트 데이터에 대한 예측 수행
test_scaled = scaler.transform(test)
test_preds = best_model.predict(test_scaled)
submission['Income'] = test_preds

# 제출 파일 생성
dt = datetime.datetime.now()
submission.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_rmse_{ensemble_rmse:4}.csv",index=False)

