import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings(action='ignore')

class CustomXGBClassifier(XGBClassifier):
    def __str__(self):
        return 'XGBClassifier()'

# aaa = CustomXGBClassifier()
# print(aaa)

#1. 데이터 
x,y = load_iris(return_X_y=True) 
print(x.shape, y.shape) # (150, 4) (150,)

datasets = load_iris()
x = datasets.data
y = datasets['target']

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
    
import matplotlib.pyplot as plt
import numpy as np

# def plot_feature_importandes_dataset(model):
#     n_features =datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)
#     plt.title(model)
    
# plot_feature_importandes_dataset(model)
# plt.show()

from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()

