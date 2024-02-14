# ======================== GradientBoostingClassifier(random_state=777) =======================
# GradientBoostingClassifier(random_state=777)
# acc :  0.9666666666666667
# [0.00100835 0.02335469 0.62409654 0.35154042]

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np

class CustomXGBClassifier(XGBClassifier):
    def __str__(self):
        return 'XGBClassifier()'

# aaa = CustomXGBClassifier()
# print(aaa)

#1. 데이터 
datasets = load_iris()
x = datasets.data
y = datasets.target

### 넘파이 삭제
# x = np.delete(x, 0, axis=1)
# print(x)    # [3.5 1.4 0.2] 1열 삭제

### 판다스 삭제
# pd.DataFrame
# 컬럼명 : datasets.feature_names

# 실습 
# 피쳐임포턴스 전체 중요도에서 하위 20~25% 컬럼들을 제거하여 데이터셋 재구성후 
# 기존 모델 결과와 비교
feature_names = datasets.feature_names
x = pd.DataFrame(x, columns=feature_names)
print(x)  
x = x.drop(['sepal length (cm)'], axis=1)
print(x)  

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    random_state=123,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    )

#2. 모델 구성
model1 = DecisionTreeClassifier(random_state=777)
model2 = RandomForestClassifier(random_state=777)
model3 = GradientBoostingClassifier(random_state=777)
model4 = CustomXGBClassifier(random_state=777)

models = [model1, model2, model3, model4]

for model in models :
    model.fit(x_train, y_train)
    print("========================", model, '=======================')
    print(model)
    print("acc : ", model.score(x_test, y_test))
    print(model.feature_importances_)
    
# ======================== DecisionTreeClassifier(random_state=777) =======================
# DecisionTreeClassifier(random_state=777)
# acc :  1.0
# [0.02506789 0.06761888 0.90731323]
# ======================== RandomForestClassifier(random_state=777) =======================
# RandomForestClassifier(random_state=777)
# acc :  0.9666666666666667
# [0.1207574  0.49170533 0.38753727]
# ======================== GradientBoostingClassifier(random_state=777) =======================
# GradientBoostingClassifier(random_state=777)
# acc :  0.9666666666666667
# [0.02361472 0.59932948 0.3770558 ]
# ======================== XGBClassifier() =======================
# XGBClassifier()
# acc :  0.9666666666666667
# [0.0191905 0.7982998 0.1825097]


