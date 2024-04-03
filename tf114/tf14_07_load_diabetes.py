# 회귀
# 7. load_diabetes
# 8. california
# 9. dacon_따릉이
# 10. kaggle bike

import tensorflow as tf
tf.compat.v1.set_random_seed(222)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1.데이터
x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape) # (442, 10) (442,)
y = y.reshape(-1,1) # (442,1) # 웨이트와 행렬연산을 해야하기 때문

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=222,
)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().fit(x)
x_data = scaler.transform(x)
x_test = scaler.transform(x_test)


xp = tf.compat.v1.placeholder(tf.float32, shape=[None,10])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1], name='weight', dtype=tf.float64))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias', dtype=tf.float64))


#2.모델
hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))
loss = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


epochs=1500
for step in range(epochs):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                 feed_dict = {xp: x_train, yp: y_train}) 
    if step % 20 ==0:
        print(step, loss_val, w_val, b_val)

y_pred = sess.run(hypothesis, feed_dict={xp: x_test})
print('예측값:', y_pred)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"MSE = {mse:.4f}")
print(f"R2 score = {r2:.4f}")

sess.close()




