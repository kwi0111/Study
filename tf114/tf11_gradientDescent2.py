
import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

#1. 데이터
x_train = [1,2,3]
y_train = [1,2,3]
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')
b = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')

#2. 모델
hypothesis = x * w + b  # 선형 방정식

#3-1. 컴파일, // model.compile(loss = 'mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))    # 예측된 거에서 y를 빼서 제곱한거의 평균 MSE

################################ optimizer start ##################################
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)  경사 하강법
# train = optimizer.minimize(loss)  
lr = 0.1
gradient = tf.reduce_mean((x * w + b - y) * x)  # 손실 함수의 w에 대한 기울기를 계산 (각 데이터 포인트에 대한 손실의 기울기를 구하고, 그 평균을 내서 전체 기울기를 얻는 과정)

descent = w -lr * gradient  # 새로운 가중치를 계산 (현재 가중치 - 계산된 기울기와 학습률)
update = w.assign(descent)  # 새 가중치를 실제 가중치 변수에 할당해 업데이트
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
