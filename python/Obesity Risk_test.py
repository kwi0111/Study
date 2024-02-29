import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, AveragePooling1D, Flatten, Conv2D, LSTM, Bidirectional, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

path = "c:/_data/kaggle/비만/"
train = pd.read_csv(path + "train.csv", index_col=0)
test = pd.read_csv(path + "test.csv", index_col=0)
sample = pd.read_csv(path + "sample_submission.csv")
x = train.drop(['NObeyesdad'], axis=1)
y = train['NObeyesdad']

TRAINSIZE = 0.8
RS = 7
NUM = 41

# 초기값 설정
best_accuracy = 0.0
best_model = None

lb = LabelEncoder()

y = lb.fit_transform(train['NObeyesdad'])

columns_to_encode = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])

for column in columns_to_encode:
    lb.fit(test[column])
    test[column] = lb.transform(test[column])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=TRAINSIZE, random_state=RS, stratify=y)

# 반복하여 모델을 훈련하고 최고의 정확도를 갖는 모델을 찾음
for i in range(1000):  # 1000번 시도
    r = np.random.randint(1, 1000)
    random_state = r

    xgb_params = {
        'objective': 'multi:softmax',
        'num_class': 7,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'random_state': random_state        
        }   
    

    model = XGBClassifier(**xgb_params)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

    print(f"Iteration {i+1}: Accuracy = {accuracy:.4f}, Best Accuracy = {best_accuracy:.4f}")

    if best_accuracy >= 0.91:
        print("Best accuracy achieved. Stopping the iterations.")
        break

if best_model is not None:
    SAVENAME = f'biman_xgb_{NUM}_best_accuracy_{best_accuracy:.4f}_r_{r}'
    best_model.save_model(f"c:\_data\_save\\{SAVENAME}.h5")

    y_submit = best_model.predict(test)
    sample['NObeyesdad'] = y_submit
    sample.to_csv(path + f"{SAVENAME}.csv", index=False)

print("Final Best Accuracy:", best_accuracy)