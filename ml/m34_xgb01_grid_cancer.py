from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn. model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#1.데이터
x, y = load_breast_cancer(return_X_y=True) 

x_train, x_test, y_train, y_test = train_test_split(x, y , random_state=123, train_size=0.85,
                                                    stratify=y,
                                                    )

scaler = MinMaxScaler()
# scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123123)
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123123)

'''
'n_estimators' : [100,200,300,400,500,1000] 디폴트 100 / 1~inf / 정수 // 경사 하강법을 이용하여 트리의 가중치 업데이트
'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] 디폴트 0.3 / 0~1 / eta // 통상적으로 작으면 작을수록 좋다. // 가중치를 업데이트하는 속도를 제어
'max_depth' : [None, 2, 3, 4, 5, 6, 7,8, 9,10] 디폴트6 / 0~inf / 정수 // 트리 계열에서 깊이
'gamma' : [0,1,2,3,4,5,6,10,100] 디폴트0 / 0~inf 
'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100] 디폴트 1 / 0~inf
'subsample' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1 ] 디폴트 1 / 0~1
'colsample_bytree' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1 ] 디폴트 1 / 0~1
'colsample_bylevel' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1 ] 디폴트 1 / 0~1
'colsample_bynode' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1 ] 디폴트 1 / 0~1
'reg_alpha' : [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 0 / 0~inf / L1 절대값 가중치 규제 / alpha // 가중치 제한
'reg_lambda' : [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 1 / 0~inf / L2 제곱 가중치 규제 / lambda 

    "n_estimators": 2000,                  # 부스팅 라운드 수 증가 / 앙상블 학습 방법
    "learning_rate": 0.05,                 # 학습률 증가
    "colsample_bylevel": 0.8,              # 레벨별 컬럼 샘플링 비율 조정
    "min_child_samples": 5,                # 최소한의 자식 샘플 수
    "bootstrap_type": "Bernoulli",         # 부트스트랩 유형 변경
    "subsample": 0.9,                      # 샘플 선택 비율 증가
    "depth": 10,                           # 트리의 최대 깊이 증가
    "border_count": 254,                   # 범주형 변수 수에 대한 근사값 증가
    "l2_leaf_reg": 7,                      # L2 정규화 강도 조정
    "leaf_estimation_iterations": 30,      # 잎 노드의 초기 추정 횟수 증가
    "leaf_estimation_method": "Gradient",  # 잎 노드의 초기 추정 방법 변경
    "thread_count": -1,                    # 사용할 스레드 수 (-1은 가능한 모든 스레드 사용)
'''

parameters = {
    'n_estimators': [100],
    'learning_rate': [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001],
    'max_depth': [3],
}

#2.모델
xgb = XGBClassifier(random_state=123)
model = RandomizedSearchCV(xgb, parameters, cv=kfold, random_state=123123,
                           n_jobs=22)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측 // train의 결과 우리가 판단할것은 test // score로 확인해야함 // 깊게 믿지말것
print("최상의 매개변수 : ", model.best_estimator_)  # 
print("최상의 매개변수 : ", model.best_params_) 

print("최상의 점수 : ", model.best_score_)  # 기준 

results = model.score(x_test, y_test)
print("최고의 점수 : ", results)  


# 최상의 매개변수 :  {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.3}
# 최상의 점수 :  0.9648625429553265
# 최고의 점수 :  1.0

