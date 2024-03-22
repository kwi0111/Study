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

x = train_csv.drop(['person_id', 'login', 'Sex', 'email_type', 'apple_rat'], axis=1)
y = train_csv['login']

# x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=RANDOM_STATE)

def objectiveRF(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),  # 조정: 트리의 수를 200에서 1000 사이로 확장
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 10, 100),  # 조정: 최대 깊이를 더 넓은 범위로 조정하여 더 깊은 트리 허용
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),  # 조정: 분할에 필요한 최소 샘플 수 범위 조정
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),  # 조정: 리프에 필요한 최소 샘플 수 조정
        # 'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0, 0.05),  # 복원: 리프의 최소 가중치 비율 조정
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),  # 조정: 최대 특성 수 결정
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 100, 2000),  # 복원 및 조정: 최대 리프 노드 수를 100에서 1000 사이로 설정
        # 'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.1),  # 조정: 불순도 감소량의 최소값 조정
        # 'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),  # 조정: 부트스트랩 샘플링을 사용할지 여부
        'random_state': RANDOM_STATE
    }

# 최고점
# 110,gini,409,11,9,0.0003708462079350574,auto,398,5.396924281913153e-06,True // 

    # 여기는 이전과 동일
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
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

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=80, n_warmup_steps=200, interval_steps=1))
study.optimize(objectiveRF, n_trials=15000, timeout=25000)

best_params = study.best_params

optuna.visualization.plot_param_importances(study)      # 파라미터 중요도 확인 그래프
optuna.visualization.plot_optimization_history(study)   # 최적화 과정 시각화

model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE)
model.fit(x, y)
preds = model.predict_proba(x)[:, 1]
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
Feature 0 (Sex): 0.0187113169480082
Feature 1 (past_login_total): 0.15659441262899576
Feature 2 (past_1_month_login): 0.40874237416118
Feature 3 (past_1_week_login): 0.26601999712420626
Feature 4 (sub_size): 0.06180898228593933
Feature 5 (email_type): 0.02256050583405757
Feature 6 (phone_rat): 0.044797342153462386
Feature 7 (apple_rat): 0.020765068864150492


'''
