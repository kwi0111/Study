import tensorflow as tf
tf.set_random_seed(777)

#1. 데이터
x = [1,2,3,4,5]
y = [1,2,3,4,5]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

#2.모델구성
hypothesis = x * w + b

#3-1. 컴파일, 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

#4. model.fit
epochs = 1000
for step in range(epochs):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))
sess.close()