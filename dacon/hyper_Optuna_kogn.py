import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score 
import optuna
from collections import OrderedDict
import datetime

dt = datetime.datetime.now()
path = 'C:\\_data\\dacon\\hyper\\'
SEED = 42
train_csv = pd.read_csv(path + "train.csv", index_col=0)
X = train_csv.drop('login', axis=1)
y = train_csv['login']

X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=SEED)


#def : 8595
def objectiveRF(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 50),  # 조정: 트리의 수를 200에서 1000 사이로 확장
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth',3, 25),  # 조정: 최대 깊이를 더 넓은 범위로 조정하여 더 깊은 트리 허용
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),  # 조정: 분할에 필요한 최소 샘플 수 범위 조정
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 13),  # 조정: 리프에 필요한 최소 샘플 수 조정
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0, 0.3),  # 복원: 리프의 최소 가중치 비율 조정
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),  # 조정: 최대 특성 수 결정
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 500),  # 복원 및 조정: 최대 리프 노드 수를 100에서 1000 사이로 설정
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.1),  # 조정: 불순도 감소량의 최소값 조정
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),  # 조정: 부트스트랩 샘플링을 사용할지 여부
        'random_state': SEED
    }
    
    skf = KFold(5, shuffle=True, random_state=SEED)
    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
    
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        predictions = model.predict_proba(X_val)[:, 1]  
        cv_scores[idx] = roc_auc_score(y_val, predictions)
    return np.mean(cv_scores)

study = optuna.create_study(direction="maximize")
# study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objectiveRF, n_trials=20000, timeout=18000)
best_params = study.best_trial.params

print(best_params)
model = RandomForestClassifier(**best_params, random_state=SEED)
model.fit(X, y)
predictions = model.predict_proba(X_test)[:, 1]  
score = roc_auc_score(y_test, predictions)
print("ROC AUC score : ", score)

# param_order = [
#     'n_estimators',
#     'criterion',
#     'max_depth',
#     'min_samples_split',
#     'min_samples_leaf',
#     'min_weight_fraction_leaf',
#     'max_features',
#     'max_leaf_nodes',
#     'min_impurity_decrease',
#     'bootstrap',
# ]
# best_params_ordered = OrderedDict({k: best_params.get(k, None) for k in param_order})
# best_params_ordered['max_depth'] = 100  
# best_params_ordered['min_weight_fraction_leaf'] = 0.0
# best_params_ordered['max_leaf_nodes'] = None  

# if score > 0.84:
#     submission = pd.DataFrame([best_params_ordered])
#     submission.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}score{score:4}.csv",index=False)

submission_csv = pd.read_csv('C:\\_data\\dacon\\hyper\\sample_submission.csv')
for label in submission_csv:
    if label in best_params.keys():
        submission_csv[label] = best_params[label]
submission_csv.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_SCORE_{score:4}.csv",index=False)
'''
0.809924809100826. -> ROC AUC score :  0.8507473730945687
0.8095118435405952. -> ROC AUC score :  0.8587390853929259

'''