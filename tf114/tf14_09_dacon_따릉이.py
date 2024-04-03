import tensorflow as tf
tf.compat.v1.set_random_seed(222)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

#1. 데이터 
path = "c:\\_data\\dacon\\ddarung\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0) 
submission_csv = pd.read_csv(path + "submission.csv")       


############## 결측치 처리, 1.제거 ############
# print(train_csv.isnull().sum())       
print(train_csv.isna().sum())           # 데이터 프레임 결측치 확인
train_csv = train_csv.dropna()          # 결측치있으면 행이 삭제됨
print(train_csv.isna().sum())           # train 결측치 삭제 후 확인
print(train_csv.info())
print(train_csv.shape)                  # (1328, 10) 

############## 결측치 처리, 2.채움 ############
test_csv = test_csv.fillna(test_csv.mean())     # test 결측치를 평균인 중간값으로 채움.

############ x 와 y를 분리 ################
x = train_csv.drop(['count'], axis=1)  
y = train_csv['count']
y = y.values.reshape(-1,1)
print(x.shape, y.shape) 

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=222,
)



xp = tf.compat.v1.placeholder(tf.float64, shape=[None,9])
yp = tf.compat.v1.placeholder(tf.float64, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([9,1], name='weight', dtype=tf.float64))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias', dtype=tf.float64))

#2.모델
hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
# loss = tf.reduce_sum(tf.square(hypothesis-y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


epochs=1000
for step in range(epochs):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                 feed_dict = {xp: x_train, yp: y_train}) 
    if step % 20 ==0:
        print(step, loss_val, w_val, b_val)

y_pred = sess.run(hypothesis, feed_dict={xp: x})
print('예측값:', y_pred)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Print the results
print(f"MSE = {mse:.4f}")
print(f"R2 score = {r2:.4f}")

sess.close()




