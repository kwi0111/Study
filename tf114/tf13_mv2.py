import tensorflow as tf
tf.compat.v1.set_random_seed(337)

#1. 데이터
x_data = [[73,51,65],           # (5,3)
          [92,98,11],
          [89,31,33],
          [99,33,100],
          [17,66,79]]
y_data = [[152],[185],[180],[205],[142]]    # (5,1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,3])    # 행무시 열우선
# input_shape = (3, )
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1]), name = 'weight')  # w에 맞게 행렬 지정해야줘야함
# (N, 3),(3, 1) = (N,1)
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name = 'bias')   # y.shape이랑 동일하게


#2  모델
# hypothesis = x * w + b
hypothesis = tf.compat.v1.matmul(x, w) + b  # 텐서1에서는 행렬곱을 생각해야한다.

#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # MSE

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5) # 0.00001
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())   # 변수 초기화

epochs=1001
for step in range(epochs):
    cost_val, _ = sess.run( [loss,train],
                                 feed_dict = {x:x_data, y: y_data})   #update와 loss변화량과 w변화량을 보겠다
    if step % 20 ==0:
        print(step, 'loss : ', cost_val)

    # _,loss_v,w_v = sess.run([update, loss, w], feed_dict = {x: x_train, y: y_train}) #update는 안보고 loss변화량과 w변화량만 보겠다
sess.close()




# #3-1 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y)) # MSE

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5) # 0.00001
# train = optimizer.minimize(loss)

# #3-2 훈련
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())   # 변수 초기화

# w_history = []
# loss_history = []

# for step in range(3000):
#     _, loss_v, w_v =sess.run([train, loss, w], feed_dict={x:x_data, y:y_data})
#     print(step, '\t', loss_v, '\t', w_v)
    
#     w_history.append(w_v)
#     loss_history.append(loss_v)


# from sklearn.metrics import r2_score, mean_absolute_error
# y_predict = sess.run(hypothesis, feed_dict={x:x_data})
# print(y_predict)

# r2 = r2_score(y_data, y_predict)
# print("R2 score : ", r2)

# mae = mean_absolute_error(y_test, y_predict)
# print("MAE : ", mae)



