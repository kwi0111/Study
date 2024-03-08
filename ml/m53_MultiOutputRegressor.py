import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# y값이 여러가지 있을때 

#1. 데이터 
x, y = load_linnerud(return_X_y=True)
print(x.shape, y.shape) # (20, 3) (20, 3)

# 최종값 -> x : [ 2, 110, 43.], y : [138, 33, 68.]

#234 모델, 훈련, 평가
model = RandomForestRegressor()
model.fit(x, y)
# score = model.score(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
                    round(mean_absolute_error(y, y_pred), 4),)  # RandomForestRegressor 스코어 :  3.518
print(model.predict([[2,110,43]]))  # [[151.67  34.02  64.52]]

model = Lasso()
model.fit(x, y)
# score = model.score(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
                    round(mean_absolute_error(y, y_pred), 4),)  # Lasso 스코어 :  7.4629
print(model.predict([[2,110,43]]))  # [[186.96683821  36.71930139  55.40868452]]

model = Ridge()
model.fit(x, y)
# score = model.score(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
                    round(mean_absolute_error(y, y_pred), 4),)  # Ridge 스코어 :  7.4569
print(model.predict([[2,110,43]]))  # [[187.32842123  37.0873515   55.40215097]]

model = XGBRegressor()
model.fit(x, y)
# score = model.score(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
                    round(mean_absolute_error(y, y_pred), 4),)  # XGBRegressor 스코어 :  0.0008
print(model.predict([[2,110,43]]))  # [[138.0005    33.002136  67.99897 ]]


# model = LGBMRegressor() # 에러 뜬다 // 컬럼이 여러개짜리 안먹힌다.
# model.fit(x, y)
# # score = model.score(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, '스코어 : ',
#                     round(mean_absolute_error(y, y_pred), 4),)  # 
# print(model.predict([[2,110,43]]))  # 
# # ValueError: y should be a 1d array, got an array of shape (20, 3) instead.

model = MultiOutputRegressor(LGBMRegressor())
model.fit(x, y)
# score = model.score(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
                    round(mean_absolute_error(y, y_pred), 4),)  # MultiOutputRegressor 스코어 :  8.91
print(model.predict([[2,110,43]]))  # [[178.6  35.4  56.1]]


model = MultiOutputRegressor(CatBoostRegressor(verbose=0))
model.fit(x, y)
# score = model.score(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
                    round(mean_absolute_error(y, y_pred), 4),)  # MultiOutputRegressor 스코어 :  0.2154
print(model.predict([[2,110,43]]))  # [[138.97756017  33.09066774  67.61547996]]


model =CatBoostRegressor(verbose=0, loss_function='MultiRMSE')  # 파라미터 찾아봐라. 
model.fit(x, y)
# score = model.score(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
                    round(mean_absolute_error(y, y_pred), 4),)  # CatBoostRegressor 스코어 :  0.0638
print(model.predict([[2,110,43]]))  # [[138.21649371  32.99740595  67.8741709 ]]
