from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedStratifiedKFold, GroupKFold 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
import tensorflow as tf
import pandas as pd
import numpy as np
import optuna
import random
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel


RANDOM_STATE = 42  
tf.random.set_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

path = 'C:\\_data\\dacon\\hyper\\'
train_csv = pd.read_csv(path + 'train.csv')

x = train_csv.drop(['person_id', 'login'], axis=1)
y = train_csv['login']

# x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=RANDOM_STATE)

def objectiveRF(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),  # 범위 조정
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 5, 20),  # 범위 조정
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 5),  # 범위 조정
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 3),  # 범위 조정
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.01, 0.04),  # 범위 조정
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 200, 500),  # 범위 조정
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0, 0.005),  # 범위 조정
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': RANDOM_STATE
    }
    
# 몇개 그냥 주석처리해서 하는게 성능이 더 높다.


    # Stratified k-fold 교차 검증을 위한 설정
    skf = KFold(n_splits=5, shuffle=True,  random_state=RANDOM_STATE)
    scores = []
    for train_idx, valid_idx in skf.split(x, y):
        x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = RandomForestClassifier(**param)
        model.fit(x_train, y_train)
        preds = model.predict_proba(x_valid)[:, 1]
        score = roc_auc_score(y_valid, preds)
        scores.append(score)

    return np.mean(scores)

from optuna.pruners import MedianPruner

# 옵튜나 얼리스탑
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1))
# study = optuna.create_study(direction="maximize", pruner=MedianPruner())
# study = optuna.create_study(direction='maximize')
study.optimize(objectiveRF, n_trials=200)

best_params = study.best_params

optuna.visualization.plot_param_importances(study)      # 파라미터 중요도 확인 그래프
optuna.visualization.plot_optimization_history(study)   # 최적화 과정 시각화

model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE)
selector = SelectFromModel(model, threshold=-np.inf, max_features=8)
selector.fit(x, y)
x_selected = selector.transform(x)


model.fit(x_selected, y)
preds = model.predict_proba(x_selected)[:, 1]
final_score = roc_auc_score(y, preds)

pred_list = model.predict_proba(x)[:,1]
auc = roc_auc_score(y,pred_list)
print(f'Final AUC score: {final_score}')
print("AUC:  ",auc)
print(best_params)

#  Trial 99 finished with value: 0.9007633587786259 and parameters: {'n_estimators': 879, 'criterion': 'gini', 'bootstrap': False, 'max_depth': 29, 'min_samples_split': 0.19835986327450797, 'min_samples_leaf': 0.2229979795940108, 'min_weight_fraction_leaf': 0.3422843335377289}. Best is trial 89 with value: 0.9122137404580153.
# {'n_estimators': 662, 'criterion': 'gini', 'bootstrap': False, 'max_depth': 32, 'min_samples_split': 0.3746056675007304, 'min_samples_leaf': 0.029695340582856743, 'min_weight_fraction_leaf': 0.04368839461045142}

submission_csv = pd.read_csv('C:\\_data\\dacon\\hyper\\sample_submission.csv')
for label in submission_csv:
    if label in best_params.keys():
        submission_csv[label] = best_params[label]
    
import datetime
dt = datetime.datetime.now()
submission_csv.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_AUC_{auc:4}.csv",index=False)

'''
n_estimators:
기본값: 10
범위: 10 ~ 1000 사이의 양의 정수. 일반적으로 값이 클수록 모델 성능이 좋아지지만, 계산 비용과 시간도 증가합니다.
criterion:
기본값: 'gini'
옵션: 'gini', 'entropy'. 'gini'는 진니 불순도를, 'entropy'는 정보 이득을 기준으로 합니다.
max_depth:
기본값: None
범위: None 또는 양의 정수. None으로 설정하면 노드가 모든 리프가 순수해질 때까지 확장됩니다. 양의 정수를 설정하면 트리의 최대 깊이를 제한합니다.
min_samples_split:
기본값: 2
범위: 2 이상의 정수 또는 0과 1 사이의 실수 (비율을 나타냄, (0, 1] ). 내부 노드를 분할하기 위해 필요한 최소 샘플 수를 지정합니다.
min_samples_leaf:
기본값: 1
범위: 1 이상의 정수 또는 0과 0.5 사이의 실수 (비율을 나타냄, (0, 0.5] ). 리프 노드가 가져야 하는 최소 샘플 수를 지정합니다.
min_weight_fraction_leaf:
기본값: 0.0
범위: 0.0에서 0.5 사이의 실수. 리프 노드에 있어야 하는 샘플의 최소 가중치 비율을 지정합니다.
max_features:
기본값: 'auto'
옵션: 'auto', 'sqrt', 'log2', None 또는 양의 정수/실수. 최적의 분할을 찾기 위해 고려할 특성의 수 또는 비율을 지정합니다. 'auto'는 모든 특성을 사용함을 의미하며, 'sqrt'와 'log2'는 각각 특성의 제곱근과 로그2를 사용합니다. None은 'auto'와 동일하게 모든 특성을 의미합니다.
max_leaf_nodes:
기본값: None
범위: None 또는 양의 정수. 리프 노드의 최대 수를 제한합니다. None은 무제한을 의미합니다.
min_impurity_decrease:
기본값: 0.0
범위: 0.0 이상의 실수. 노드를 분할할 때 감소해야 하는 불순도의 최소량을 지정합니다.
bootstrap:
기본값: True
옵션: True, False. True는 부트스트랩 샘플을 사용하여 개별 트리를 학습시킵니다. False는 전체 데이터셋을 사용하여 각 트리를 학습시킵니다.

score:  0.9167303284950343
AUC:   0.8193942213689205
0.7669699367

{'n_estimators': 595, 'criterion': 'gini', 'max_depth': 97, 'min_samples_split': 5, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.009437059428179115, 'max_features': 'log2', 'max_leaf_nodes': 277, 'min_impurity_decrease': 0.0005788070004602987, 'bootstrap': True}
score:  0.9243697478991597
AUC:   0.8494564129141686
0.7722310127

AUC:   0.8562026643423362
0.7792721519

AUC:   0.8527220579747701

Final AUC score: 0.8704195573563881
AUC:   0.8704195573563881
{'n_estimators': 43, 'criterion': 'gini', 'max_depth': 28, 'min_samples_split': 7, 'min_samples_leaf': 5, 'min_weight_fraction_leaf': 0.02025656243386623, 'max_features': 'log2', 'bootstrap': True}



'''
