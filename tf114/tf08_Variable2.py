import tensorflow as tf
tf.compat.v1.set_random_seed(777)


#1.데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]

x = tf.compat.v1.placeholder(tf.float32, shape=[None]) 
y = tf.compat.v1.placeholder(tf.float32, shape=[None]) 

w = tf.Variable(tf.random_normal([1]),dtype=tf.float32)  # 정규 분포
b = tf.Variable(tf.random_normal([1]),dtype=tf.float32)  

# sess = tf.compat.v1.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(w)) #  [2.2086694]

#2.모델구성
hypothesis = x * w + b  # predict 예측값.

#3. 컴파일, 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))    # MSE
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0823)   # 그냥 경사하강법 // 로스 최소화
train = optimizer.minimize(loss)   # 손실 함수를 최소화하는 연산
# model.compile(loss='mse', optimizer ='sgd')

#3-2. 훈련
# 변수 초기화
# sess = tf.compat.v1.Session()
# with tf.compat.v1.Session() as sess:
#     sess.run(tf.global_variables_initializer()) # 변수 2개 초기화
    
#     epochs = 101
#     for step in range(epochs):
#         _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
#                                              feed_dict={x:x_data, y:y_data})   # 결과치 4개 나온다., 언더바로 위치 표시만해준다.
#         if step % 1 == 0:  # 20번마다 한번씩 보여주겠다 .-> verbose
#             print(step+1, loss_val, w_val, b_val)

#     x_data = [6, 7, 8]
#     x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
#     y_pred = x_test * w_val + b_val
#     print('[6, 7, 8] 예측:', sess.run(hypothesis, feed_dict={x: x_data}))

# [실습]
# 07_2를 카피해서 아래를 맹그러봐!!!

################ 1. Session() // sess.run(변수) ###################
# with tf.compat.v1.Session() as sess:
#     sess.run(tf.global_variables_initializer()) # 변수 2개 초기화
    
#     epochs = 101
#     for step in range(epochs):
#         _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
#                                              feed_dict={x:x_data, y:y_data})   # 결과치 4개 나온다., 언더바로 위치 표시만해준다.
#         if step % 1 == 0:  # 20번마다 한번씩 보여주겠다 .-> verbose
#             print(step+1, loss_val, w_val, b_val)

#     x_data = [6, 7, 8]
#     x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
#     y_pred = x_test * w_val + b_val
#     print('[6, 7, 8] 예측:', sess.run(hypothesis, feed_dict={x: x_data}))


################ 2. Session() // 변수.eval(session=sess) ###################
# with tf.compat.v1.Session() as sess:
#     sess.run(tf.global_variables_initializer()) # 변수 2개 초기화
    
#     epochs = 101
#     for step in range(epochs):
#         _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
#                                              feed_dict={x:x_data, y:y_data})   # 결과치 4개 나온다., 언더바로 위치 표시만해준다.
#         if step % 1 == 0:  # 20번마다 한번씩 보여주겠다 .-> verbose
#             print(step+1, loss_val, w_val, b_val)

#     x_data = [6, 7, 8]
#     x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
#     y_pred = x_test * w_val + b_val
#     y_pred = y_pred.eval(feed_dict = {x_test:x_data})
#     # print('[6, 7, 8] 예측:', sess.run(hypothesis, feed_dict={x: x_data}))
#     print('y_pred2 :', y_pred)
#     sess.close()



################ 3. InteractiveSession() // sess.run(변수) ###################
sess= tf.compat.v1.InteractiveSession()
sess.run(tf.global_variables_initializer()) # 변수 2개 초기화

epochs = 101
for step in range(epochs):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                            feed_dict={x:x_data, y:y_data})   # 결과치 4개 나온다., 언더바로 위치 표시만해준다.
    if step % 1 == 0:  # 20번마다 한번씩 보여주겠다 .-> verbose
        print(step+1, loss_val, w_val, b_val)

x_data = [6, 7, 8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_pred3 = x_test * w_val + b_val
y_pred3 = y_pred3.eval(feed_dict = {x_test:x_data})
# print('[6, 7, 8] 예측:', sess.run(hypothesis, feed_dict={x: x_data}))
print('y_pred3 :', y_pred3)
sess.close()

