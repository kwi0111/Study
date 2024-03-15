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
    'n_estimators': trial.suggest_int('n_estimators', 100, 1000, 50),  # 100에서 1000까지 50 단위로 트리의 개수를 선택
    'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),  # 지니 불순도 또는 엔트로피를 사용할지 선택
    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),  # 부트스트랩 샘플링을 사용할지 여부 선택
    'max_depth': trial.suggest_int('max_depth', 4, 32),  # 트리의 최대 깊이를 4에서 32까지 선택
    'random_state': RANDOM_STATE,  # 랜덤 시드 값으로 고정
    'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),  # 노드를 분할하기 위한 최소 샘플 수를 선택
    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 100),  # 리프 노드가 가져야 하는 최소 샘플 수를 선택
    'min_weight_fraction_leaf': trial.suggest_uniform('min_weight_fraction_leaf', 0, 0.5),  # 리프 노드의 최소 가중치 비율을 선택
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