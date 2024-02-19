from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv   # 아직 정식 버젼 x
from sklearn. model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#1.데이터
x, y = load_diabetes(return_X_y=True) 

x_train, x_test, y_train, y_test = train_test_split(x, y , random_state=123, train_size=0.85,
                                                    # stratify=y,
                                                    )

# scaler = MinMaxScaler()
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123123)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123123)

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
'''

parameters = {
    'n_estimators': [100],
    'learning_rate': [0.1],
    'max_depth': [5],
}

#2.모델
model = XGBRFRegressor(random_state=777)
# model = XGBClassifier(random_state=123)
# model = RandomizedSearchCV(xgb, parameters, cv=kfold, random_state=123123,
#                            n_jobs=22)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("최고의 점수1 : ", results)  



# 최상의 매개변수 :  {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.3}
# 최상의 점수 :  0.9648625429553265
# 최고의 점수 :  1.0

# 최고의 점수 :  0.5769946422706148

model.set_params(gamma=1) # 파라미터만 0.3으로 바꿈 // 튜닝할때 디폴트로 해놓고 파라미터 수정
model.fit(x_train, y_train)
results = model.score(x_test, y_test)
print("최고의 점수2 : ", results)  

model.set_params(learning_rate=0.9) 
model.fit(x_train, y_train)
results = model.score(x_test, y_test)
print("최고의 점수3 : ", results)  

model.set_params(n_estimators=100,
                 learning_rate=0.9,
                 max_depth=10,
                 reg_alpha=0,
                 reg_lambda=1,
                 min_child_weight=0.5
                 ) 
model.fit(x_train, y_train)
results = model.score(x_test, y_test)
print("최고의 점수4 : ", results)  

# print("사용 파라미터 : ", model.get_params())

