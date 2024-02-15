import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import sklearn as sk
print(sk.__version__) # 1.1.3

# class CustomXGBClassifier(XGBClassifier):
#     def __str__(self):
#         return 'XGBClassifier()'

# aaa = CustomXGBClassifier()
# print(aaa)

#1. 데이터 
datasets = load_iris()
x = datasets.data
y = datasets['target']  # 넘파이 백터 상태

df = pd.DataFrame(x, columns=datasets.feature_names)
print(df)
df['Target (Y)'] = y
print(df)

print("======================== 상관계수 히트맵 =============================")
print(df.corr())
# ======================== 상관계수 히트맵 =============================
#                    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target (Y)
# sepal length (cm)           1.000000         -0.117570           0.871754          0.817941    0.782561
# sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126   -0.426658
# petal length (cm)           0.871754         -0.428440           1.000000          0.962865    0.949035
# petal width (cm)            0.817941         -0.366126           0.962865          1.000000    0.956547
# Target (Y)                  0.782561         -0.426658           0.949035          0.956547    1.000000
# 0에 가까울수록 안맞는다 // 1에 가까울수록 맞다 // 음수 = 반대로 상관있다.
# x끼리 상관관계 높으면 과적합 의심
# y값과 상관관계 높을수록 좋음

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib 
print(sns.__version__)  # 0.12.2
print(matplotlib.__version__)  # 베이스 3.7.2 // 3.8.0은 안나옴

sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(),
            square=True,
            annot=True,     # 표안에 수치 명시
            cbar=True       # 사이드 바
            )
plt.show()
# matplotlib의 3.7.2에서는 수치가 잘나오나, 3.8.0에서는 수치가 안나왜 그래서 버전 롤백했다.
#  양수일 때는 하나의 값이 증가할 때 다른 변수의 값도 증가하는 관계, 음수일 때는 하나의 값이 증가하면 다른 변수의 값은 감소하는 관계

