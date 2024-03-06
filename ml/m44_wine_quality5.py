import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
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

############################################ 신뢰..? // 너무 많이 떨어져있는 아이들은 ACC보다 F1스코어로 봐야함
for i, v in enumerate(y):
    if v <=4:
        y[i] = 0
    elif v==5:
        y[i] = 1
    # elif v==6:
    #     y[i] = 2
    # elif v==7:
    #     y[i] = 3
    # elif v==8:
    #     y[i] = 4
    else:
        y[i] = 2

print(y.value_counts().sort_index())



'''
new_y = []  # 새로운 클래스를 저장할 리스트 초기화

for old_class in y:
    if old_class == 3:
        new_class = 0
    elif old_class in [4, 5]:
        new_class = 1
    elif old_class in [6, 7]:
        new_class = 2
    elif old_class == 8:
        new_class = 3
    elif old_class == 9:
        new_class = 4
    else:
        new_class = old_class
    new_y.append(new_class)

# 기존의 y 배열을 새로운 클래스로 업데이트
y = np.array(new_y)

# import matplotlib.pyplot as plt
# plt.boxplot(x)
# plt.show()

import matplotlib.pyplot as plt
# 품질 등급을 새로운 클래스에 매핑
new_quality_mapping = {3: 0, 4: 1, 5: 1, 6: 2, 7: 2, 8: 3, 9: 4}

# 매핑된 품질 등급에 따라 데이터 개수 계산
new_quality_counts = train_csv['quality'].map(new_quality_mapping).value_counts().sort_index()

# 시각화
plt.figure(figsize=(10, 6))
plt.bar(new_quality_counts.index, new_quality_counts.values, color='skyblue')  # 클래스를 0부터 4까지로 표시
plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Distribution of Wine Quality')
plt.xticks(new_quality_counts.index, new_quality_counts.index)  # x축 눈금을 새로운 클래스로 설정
plt.show()
'''

y = LabelEncoder().fit_transform(y)

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0

test_csv.loc[test_csv['type'] == 'red', 'type'] = 1 
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

def outliers(x):
    x_numeric = x.drop(columns=['type'])  # 'type' 열 제외하고 숫자형 데이터만 사용
    quartile_1, q2, quartile_3 = np.percentile(x_numeric,[25,50,75])
    
    print('1사분위 :', quartile_1)
    print('q2 :', q2)
    print('3사분위 :', quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr :", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)   #범위 지정 가능
    upper_bound = quartile_3 + (iqr * 1.5)
    
    # 'type' 열의 문자열 값을 숫자로 변환하여 처리
    x['type'] = LabelEncoder().fit_transform(x['type'])

    return np.where((x_numeric>upper_bound) |
                    (x_numeric<lower_bound))
    # or -> 두가지 조건 중 하나라도 만족하는게 있으면 리턴

outliers_loc = outliers(x)
print("이상치의 위치 :", outliers_loc)
# print("이상치 개수 :", num_outliers)
print((outliers_loc[0]))    #

x_train, x_test, y_train, y_test = train_test_split(
    x, y , random_state=1, train_size=0.8,
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = RandomForestClassifier()

#.3 훈련
model.fit(x_train, y_train,
          )

#4. 평가, 예측
results = model.score(x_test, y_test)
# print('최종점수 : ', results)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average='macro')
print('acc_score : ', acc)
print('f1_score : ', f1)


'''
라벨의 갯수가 다르면 ACC 신뢰할수 있는지 판단해야한다 // 라벨 불균형 -> smote 

라벨 변경 0 1 2
acc_score :  0.7990909090909091
f1_score :  0.5960878212643806
라벨 원판 
acc_score :  0.6654545454545454
f1_score :  0.39563367915649866
'''




