from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_diabetes, load_breast_cancer, load_digits

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

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


#2.모델
# model = XGBClassifier()  # mlogloss 디폴트
# model = XGBRegressor(random_state=777, **parameters)
# path = "C:/_data/_save/_pickle_test/"
# model = pickle.load(open(path + 'm39_pickle1_save.dat', 'rb'))  # 모델과 하이퍼파라미터 다 들어있다. // 'rb' 파일을 이진 읽기(read) 모드

import joblib
path = "C:/_data/_save/_joblib_test/"
model = joblib.load(path+"m40_joblib1_save.dat")


#4. 평가, 예측
# print("사용 파라미터 : ", model.get_params())
results = model.score(x_test, y_test)
y_pred = model.predict(x_test)
from sklearn.metrics import r2_score, accuracy_score, f1_score, roc_auc_score
print("r2 : ", r2_score(y_test, y_pred)) 
print("최종 점수 : ", results)  

# r2 :  0.9060469609385955
# 최종 점수 :  0.9740740740740741

##################################################################
# pickle, joblib
# import pickle
# path = "C:/_data/_save/_pickle_test/"
# pickle.dump(model, open(path + 'm39_pickle1_save.dat', 'wb'))




