from keras.datasets import mnist
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from keras.callbacks import EarlyStopping
import time


(x_train, y_train), (x_test, y_test) = mnist.load_data() # y값 필요 없어서 x값 2개만 받을거다 // _
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

# x = np.append(x_train, x_test, axis=0)  # 행단위로 붙일거
# x = np.concatenate([x_train, x_test], axis=0)  # 행단위로 붙일거
# x = x.reshape(-1, 784)

# print(x.shape)  # (70000, 28, 28)

# # scaler = StandardScaler()
# # x = scaler.fit_transform(x)
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

results = []

for n_components in [154, 331, 486, 713, 784]:
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    model = Sequential([
        Dense(512, activation='relu', input_shape=(n_components,)),
        Dense(256, activation='relu',),
        Dropout(0.01),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    start_time = time.time()
    model.fit(x_train_pca, y_train, epochs=300, batch_size=128, validation_split=0.2, verbose=1, callbacks=[early_stopping])
    end_time = time.time()

    loss, acc = model.evaluate(x_test_pca, y_test)
    elapsed_time = end_time - start_time

    results.append((n_components, elapsed_time, acc))

for result in results:
    print("PCA = {}, 걸린시간 {:.2f}초, acc = {:.4f}".format(*result))


# PCA = 154, 걸린시간 3.69초, acc = 0.9536
# PCA = 331, 걸린시간 4.01초, acc = 0.9500
# PCA = 486, 걸린시간 4.49초, acc = 0.9438
# PCA = 713, 걸린시간 4.91초, acc = 0.9488
# PCA = 784, 걸린시간 5.00초, acc = 0.9507

# PCA = 154, 걸린시간 39.85초, acc = 0.9721 날린게 제일 좋다.
# PCA = 331, 걸린시간 36.09초, acc = 0.9685
# PCA = 486, 걸린시간 42.59초, acc = 0.9723
# PCA = 713, 걸린시간 31.05초, acc = 0.9670
# PCA = 784, 걸린시간 37.06초, acc = 0.9700

'''
pca = PCA(n_components=154)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print(x_train.shape)  # (60000, 154)
print(x_test.shape)   # (10000, 154)

evr = pca.explained_variance_ratio_
print(evr)
cumsum = np.cumsum(evr)

# 모델 구성
model = Sequential([
    Dense(512, activation='relu', input_shape=(154,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# 모델 평가
results = model.evaluate(x_test, y_test)
print('acc:', results[1])



'''
# print(cumsum)
# print(np.argmax(cumsum >= 0.95) + 1)    # 154
# print(np.argmax(cumsum >= 0.99) + 1)    # 331
# print(np.argmax(cumsum >= 0.999) + 1)   # 486
# print(np.argmax(cumsum >= 1.0) + 1)     # 713

# 1. 70000,154
# 2. 70000,331
# 3. 70000,486
# 4. 70000,713
# 5. 70000,784 원본

################## 예시 #################
# 결과1. PCA=154
# 걸린시간 0000초
# acc = 0.0000

# 결과2. PCA=331
# 걸린시간 0000초
# acc = 0.0000

# 결과3. PCA=486
# 걸린시간 0000초
# acc = 0.0000

# 결과4. PCA=713
# 걸린시간 0000초
# acc = 0.0000

# 결과5. PCA=784
# 걸린시간 0000초
# acc = 0.0000









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

