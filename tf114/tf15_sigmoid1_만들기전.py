import tensorflow as tf
tf.compat.v1.set_random_seed(777)

#1. 데이터
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]] # (6,2)
y_data = [[0], [0], [0], [1], [1], [1]] # (6,1)


###########################################
##### [실습] 그냥 한번 만든다.
###########################################
x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])    # 행무시 열우선
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]), name = 'weight', dtype=tf.float32)  # w에 맞게 행렬 지정해야줘야함
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name = 'bias', dtype=tf.float32)   # y.shape이랑 동일하게

#2  모델
hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # MSE

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5) # 0.00001
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())   # 변수 초기화

epochs=1001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run( [loss,train, w ,b],
                                 feed_dict = {x:x_data, y: y_data})   #update와 loss변화량과 w변화량을 보겠다
    if step % 20 ==0:
        print(step, 'loss : ', cost_val)
        
print(w_val, b_val)    
# [[0.55259436]
#  [0.5237099 ]]      
print(type(w_val))      # 텐서플로우는 다 <class 'numpy.ndarray'>

#4. 평가, 예측
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None,2])

# y_predit = x_test * w_val + b_val # 이게아니다.
y_pred = tf.matmul(x_test, w_val) +b_val
y_predict = sess.run(y_pred, feed_dict={x_test:x_data})
print(y_predict)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_data, y_predict)
print('R2_SCORE : ' , r2) # R2_SCORE :  -27.418963522727875

mse = mean_squared_error(y_data, y_predict)
print('MSE : ' , mse)   # MSE :  7.104740880681969

sess.close()