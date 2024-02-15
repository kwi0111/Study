from keras.datasets import mnist
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# n_components = 
lda = LinearDiscriminantAnalysis() # n_classes(라벨) 의 갯수에 영향을 받는다. // 차원이 줄어든다. // 분류 부분만 쓴다.
# n_components는 n_features 또는 n_classes - 1 값보다는 커야한다.
lda.fit(x_train, y_train)  # x_train, y_train으로 수정
x_train_lda = lda.transform(x_train)
x_test_lda = lda.transform(x_test)

for n_components in [1,2,3,4,5,6,7,8,9]:
    if n_components <= min(x_train.shape[1], len(np.unique(y_train)) - 1):
        lda = LinearDiscriminantAnalysis(n_components=n_components
                                         )
        lda.fit(x_train, y_train)  # x_train, y_train으로 수정
        x_train_lda = lda.transform(x_train)
        x_test_lda = lda.transform(x_test)

        model = Sequential([
            Dense(512, activation='relu', input_shape=(n_components,)),
            Dense(256, activation='relu'),
            Dropout(0.01),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        start_time = time.time()
        model.fit(x_train_lda, y_train, epochs=300, batch_size=128, validation_split=0.2, verbose=1, callbacks=[early_stopping])
        end_time = time.time()

        loss, acc = model.evaluate(x_test_lda, y_test)
        elapsed_time = end_time - start_time

        results.append((n_components, elapsed_time, acc))
    else:
        print(f"n_components={n_components} 값은 유효하지 않습니다.")

# 결과 출력
for result in results:
    print("LDA = {}, 걸린시간 {:.2f}초, acc = {:.4f}".format(*result))


# PCA = 154, 걸린시간 3.69초, acc = 0.9536
# PCA = 331, 걸린시간 4.01초, acc = 0.9500
# PCA = 486, 걸린시간 4.49초, acc = 0.9438
# PCA = 713, 걸린시간 4.91초, acc = 0.9488
# PCA = 784, 걸린시간 5.00초, acc = 0.9507

# PCA = 154, 걸린시간 39.85초, acc = 0.9721
# PCA = 331, 걸린시간 36.09초, acc = 0.9685
# PCA = 486, 걸린시간 42.59초, acc = 0.9723
# PCA = 713, 걸린시간 31.05초, acc = 0.9670
# PCA = 784, 걸린시간 37.06초, acc = 0.9700

# LDA = 1, 걸린시간 8.55초, acc = 0.4160
# LDA = 2, 걸린시간 14.47초, acc = 0.5666
# LDA = 3, 걸린시간 15.00초, acc = 0.7509
# LDA = 4, 걸린시간 17.17초, acc = 0.8316
# LDA = 5, 걸린시간 15.72초, acc = 0.8482
# LDA = 6, 걸린시간 17.82초, acc = 0.8701
# LDA = 7, 걸린시간 13.39초, acc = 0.8942
# LDA = 8, 걸린시간 14.82초, acc = 0.9176
# LDA = 9, 걸린시간 13.31초, acc = 0.9204


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

