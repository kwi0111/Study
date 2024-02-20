from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_diabetes, load_breast_cancer, load_digits

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#1.데이터
x, y = load_digits(return_X_y=True) 

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

parameters = {
    'n_estimators': 10000,
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight' : 10,
}

#2.모델
model = XGBClassifier()  # mlogloss 디폴트
# model = XGBRegressor(random_state=777, **parameters)
model.set_params(
    early_stopping_rounds=10,
    **parameters,
                 ) 

#3. 훈련
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=15,
          )

#4. 평가, 예측
# print("사용 파라미터 : ", model.get_params())
results = model.score(x_test, y_test)
y_pred = model.predict(x_test)
from sklearn.metrics import r2_score, accuracy_score, f1_score, roc_auc_score
print("r2 : ", r2_score(y_test, y_pred)) 
print("최종 점수 : ", results)  

##################################################################
# pickle, joblib
# import pickle
# path = "C:/_data/_save/_pickle_test/"
# pickle.dump(model, open(path + 'm39_pickle1_save.dat', 'wb')) # dump 직렬화하여 저장 // 'wb' 이진 쓰기 모드 //


# import joplib
# path = "C:/_data/_save/_joplib_test/"
# joplib.dump(model, path+"m40_joplib1_save.dat")


# r2 :  0.9060469609385955
# 최종 점수1 :  0.9740740740740741

# print(xgb.__version__)
import xgboost as xgb

path = "C:/_data/_save/"
model.save_model(path+"m41_xgb1_save_model.dat")

