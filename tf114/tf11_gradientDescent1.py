
import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

#1. 데이터
x_train = [1,2,3]
y_train = [1,2,3]
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')

#2. 모델
hypothesis = x * w

#3-1. 컴파일, // model.compile(loss = 'mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))    # 예측된 거에서 y를 빼서 제곱한거의 평균 MSE

################################ optimizer start ##################################
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)   
# train = optimizer.minimize(loss)  
lr = 0.1
gradient = tf.reduce_mean((x * w - y) * x)

descent = w -lr * gradient
update = w.assign(descent)
################################ optimizer  ##################################
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

w_history = []
loss_history = []

for step in range(33):
    _, loss_v, w_v =sess.run([update, loss, w], feed_dict={x:x_train, y:y_train})
    print(step, '\t', loss_v, '\t', w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)
sess.close()


# print('==================================== W history ==========================================')
# print(w_history)
# print('=================================== LOSS history ==========================================')
# print(loss_history)


plt.plot(w_history, loss_history)
plt.xlabel('weights')
plt.ylabel('loss')
plt.show()


# # w가 30일 때의 인덱스를 찾습니다.
# w_index = w_history.index(49)

# # 해당 인덱스에 대한 손실 값을 확인합니다.
# loss_at_w_30 = loss_history[w_index]

# print("w가 30일 때의 손실:", loss_at_w_30)
