from keras.datasets import mnist
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

####################[ 실습 ]########################
# pca를 통해 0.95이상인 n_components는 몇개?
# 0.95 이상
# 0.99 이상
# 0.999 이상
# 1.0 일때 몇개?

# 힌트 np.argmax
##################################################


(x_train, _), (x_test, _) = mnist.load_data() # y값 필요 없어서 x값 2개만 받을거다 //
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

# x = np.append(x_train, x_test, axis=0)  # 행단위로 붙일거
x = np.concatenate([x_train, x_test], axis=0)  # 행단위로 붙일거
x = x.reshape(-1, 784) # (70000, 784)

print(x.shape)  # (70000, 784)

# scaler = StandardScaler()
# x = scaler.fit_transform(x)

pca = PCA(n_components=784)
x = pca.fit_transform(x) 
print(x)

evr = pca.explained_variance_ratio_
print(evr)
cumsum = np.cumsum(evr)
print(cumsum)
print(np.argmax(cumsum >= 0.95) + 1)
print(np.argmax(cumsum >= 0.99) + 1)
print(np.argmax(cumsum >= 0.999) + 1)
print(np.argmax(cumsum >= 1.0) + 1)


# n_components_95 = np.argmax(np.cumsum(evr) >= 0.95) + 1
# print("n_components for 95% explained variance ratio:", n_components_95)

# 주어진 explained variance ratio 이상인 n_components의 개수 찾기
# for target_ratio in [0.95, 0.99, 0.999, 1.0]:
#     n_components = np.argmax(np.cumsum(evr) >= target_ratio) + 1
#     print(f"n_components for {target_ratio:.3f} explained variance ratio:", n_components)

# n_components for 0.950 explained variance ratio: 332
# n_components for 0.990 explained variance ratio: 544

# n_components for 0.999 explained variance ratio: 683
# n_components for 1.000 explained variance ratio: 1

