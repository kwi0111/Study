from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
import tensorflow as tf
import pandas as pd
import numpy as np
import optuna
import random

RANDOM_STATE = 42  
tf.random.set_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

path = 'C:\\_data\\dacon\\hyper\\'
train_csv = pd.read_csv(path + 'train.csv')

x = train_csv.drop(['person_id', 'login'], axis=1)
y = train_csv['login']

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=RANDOM_STATE)

def objectiveRF(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'criterion': trial.suggest_categorical('criterion', ['gini']),  # Hyperopt 예제에서는 'gini'만 사용했으므로
        'bootstrap': trial.suggest_categorical('bootstrap', [True]),  # Hyperopt 예제에서는 True만 사용했으므로
        'max_depth': trial.suggest_int('max_depth', 5, 200),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 500),
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0, 0.2),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0, 0.5),
        # 추가되지 않은 class_weight는 기존 제안대로 사용
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
        'random_state': RANDOM_STATE  # 고정 랜덤 시드
    }
    
    # 학습 모델 생성
    model = RandomForestClassifier(**param)
    rf_model = model.fit(x_train, y_train) # 학습 진행
    
    # 모델 성능 확인
    # score = auc(rf_model.predict(x_test), y_test)
    score = model.score(x_test,y_test)
    
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objectiveRF, n_trials=2000)

best_params = study.best_params
print(best_params)

optuna.visualization.plot_param_importances(study)      # 파라미터 중요도 확인 그래프
optuna.visualization.plot_optimization_history(study)   # 최적화 과정 시각화

# model = RandomForestClassifier(**best_params)
model = RandomForestClassifier(**best_params)
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
pred_list = model.predict_proba(x_test)[:,1]
print("score: ",score)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test,pred_list)
print("AUC:  ",auc)

#  Trial 99 finished with value: 0.9007633587786259 and parameters: {'n_estimators': 879, 'criterion': 'gini', 'bootstrap': False, 'max_depth': 29, 'min_samples_split': 0.19835986327450797, 'min_samples_leaf': 0.2229979795940108, 'min_weight_fraction_leaf': 0.3422843335377289}. Best is trial 89 with value: 0.9122137404580153.
# {'n_estimators': 662, 'criterion': 'gini', 'bootstrap': False, 'max_depth': 32, 'min_samples_split': 0.3746056675007304, 'min_samples_leaf': 0.029695340582856743, 'min_weight_fraction_leaf': 0.04368839461045142}

submission_csv = pd.read_csv('C:\\_data\\dacon\\hyper\\sample_submission.csv')
for label in submission_csv:
    if label in best_params.keys():
        submission_csv[label] = best_params[label]
    
import datetime
dt = datetime.datetime.now()
submission_csv.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_acc_{score:4}.csv",index=False)
# submit_csv.to_csv(f'c:/Study/Dacon/RF_tuning/submit/AUC_{auc:.6f}.csv',index=False)