import tensorflow as tf
tf.set_random_seed(777)

#1.데이터
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111, dtype=tf.float32)  # weight
b = tf.Variable(0, dtype=tf.float32)    # bias

#2.모델구성
# y = wx + b

# hypothesis = w * x + b    # 이제는 이거 아니다. 아니었다.
hypothesis = x * w + b  # predict 예측값.

#3. 컴파일, 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))    # MSE
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)   # 그냥 경사하강법 // 로스 최소화
train = optimizer.minimize(loss)   # 손실 함수를 최소화하는 연산

# model.compile(loss='mse', optimizer ='sgd')


#3-2. 훈련
# 변수 초기화
# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 변수 2개 초기화

    #3-3 model.fit
    epochs = 4000
    for step in range(epochs):
        sess.run(train) # 1에포
        if step % 20 == 0:  # 20번마다 한번씩 보여주겠다 .-> verbose
            print(step, sess.run(loss), sess.run(w), sess.run(b))   # verbose와 model.weight에서 봤떤 애들 // 특정 간격으로 현재 단계, 손실 함수의 값, 가중치 w, 편향 b의 값을 출력
    # sess.close()  # 끈다.
# sess는 저장되어있기때문에 닫아줘야한다.
# 
