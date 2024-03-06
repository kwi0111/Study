import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier


#1. 데이터
path = "C:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


x = train_csv.drop(['quality'], axis= 1)
y = train_csv['quality']

def outliers(x):
    x_numeric = x.drop(columns=['type'])  # 'type' 열 제외하고 숫자형 데이터만 사용
    quartile_1, q2, quartile_3 = np.percentile(x_numeric,[25,50,75])
    
    print('1사분위 :', quartile_1)
    print('q2 :', q2)
    print('3사분위 :', quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr :", iqr)
    lower_bound = quartile_1 - (iqr * 1.3)   #범위 지정 가능
    upper_bound = quartile_3 + (iqr * 1.3)
    
    # 'type' 열의 문자열 값을 숫자로 변환하여 처리
    x['type'] = LabelEncoder().fit_transform(x['type'])

    outliers_indices = np.where((x_numeric>upper_bound) | (x_numeric<lower_bound))
    
    # 전체 이상치 개수
    total_outliers = len(outliers_indices[0])
    
    # 이상치 개수를 피처별로 저장
    num_outliers = [len(outliers_indices[0][outliers_indices[1] == i]) for i in range(x_numeric.shape[1])]
    
    return outliers_indices, num_outliers, total_outliers

outliers_loc, num_outliers, total_outliers = outliers(x)
print("이상치의 위치 :", outliers_loc)
print("이상치 개수 :", num_outliers)
print("전체 이상치 개수 :", total_outliers)
print((outliers_loc[0]))    

import matplotlib.pyplot as plt
quality_counts = train_csv.groupby('quality').count()['fixed acidity']   # quality 칼럼을 기준으로 groupby하여 각 품질 등급별로 데이터 개수를 계산

# 시각화
plt.figure(figsize=(10, 6))
plt.bar(quality_counts.index, quality_counts.values, color='skyblue')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Distribution of Wine Quality')
plt.xticks(quality_counts.index)
plt.show()

y = LabelEncoder().fit_transform(y)

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0

test_csv.loc[test_csv['type'] == 'red', 'type'] = 1 
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

x_train, x_test, y_train, y_test = train_test_split(
    x, y , random_state=1, train_size=0.8,
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'learning_rate': 0.13349839953884737,
                'n_estimators': 99,
                'max_depth': 8,
                'min_child_weight': 3.471164143831403e-06,
                'subsample': 0.6661302167437514,            #dropout 비슷
                'colsample_bytree': 0.9856906281904222,
                'gamma': 4.5485144879936555e-06,
                'reg_alpha': 0.014276113125688179,
                'reg_lambda': 10.121476098960851,
                # 'nthread' : 20,
                'tree_method' : 'gpu_hist',
                'predictor' : 'gpu_predictor',
                }

#2. 모델
model = XGBClassifier()
model.set_params(early_stopping_rounds=10, **parameters)

#.3 훈련
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=1,
          eval_metric='mlogloss',
          )

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score : ', acc)


'''
최종점수 :  0.6790909090909091
acc_score :  0.6790909090909091
'''




