import matplotlib.pyplot as plt
import tensorflow as tf
tf.set_random_seed(777)

#1.데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]

x = tf.compat.v1.placeholder(tf.float32, shape=[None]) 
y = tf.compat.v1.placeholder(tf.float32, shape=[None]) 

# w = tf.Variable(111, dtype=tf.float32)  # weight
# b = tf.Variable(0, dtype=tf.float32)    # bias
w = tf.Variable(tf.random_normal([1]),dtype=tf.float32)  # 정규 분포
b = tf.Variable(tf.random_normal([1]),dtype=tf.float32)  

# sess = tf.compat.v1.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(w)) #  [2.2086694]



#2.모델구성
# y = wx + b

# hypothesis = w * x + b    # 이제는 이거 아니다. 아니었다.
hypothesis = x * w + b  # predict 예측값.

#3. 컴파일, 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))    # MSE
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0823)   # 그냥 경사하강법 // 로스 최소화
train = optimizer.minimize(loss)   # 손실 함수를 최소화하는 연산

# model.compile(loss='mse', optimizer ='sgd')


#3-2. 훈련
# 변수 초기화
# sess = tf.compat.v1.Session()
loss_val_list = []
w_val_list = []

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 변수 2개 초기화
    epochs = 101
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                             feed_dict={x:x_data, y:y_data})   # 결과치 4개 나온다., 언더바로 위치 표시만해준다.
        if step % 1 == 0:  # 20번마다 한번씩 보여주겠다 .-> verbose
            # print(step, sess.run(loss), sess.run(w), sess.run(b))   # verbose와 model.weight에서 봤떤 애들 // 특정 간격으로 현재 단계, 손실 함수의 값, 가중치 w, 편향 b의 값을 출력
            print(step+1, loss_val, w_val, b_val)
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
        
        
        print('______--------_________---------'*10)
        #4. 예측 model.predict()
        ######################## [실습] ###########################
        x_pred_data= [6, 7 ,8]
        x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
        ###########################################################
        # y_predict = xw + b
        #1. 파이썬 방식
        y_predict = x_pred_data * w_val + b_val
        print('[6,7,8]의 예측 : ', y_predict)

        #2. placeholder에 넣어서 
        y_preidct2 = x_test * w_val + b_val
        print('[6,7,8]의 예측 : ', sess.run(y_preidct2, feed_dict={x_test:x_pred_data}))

print(loss_val_list)
print('==========='*10)
print(w_val)
print('==========='*10)


# plt.plot(loss_val_list)
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()

# plt.plot(w_val_list)
# plt.xlabel('epochs')
# plt.ylabel('weights')
# plt.show()

# plt.scatter(w_val_list, loss_val_list)
# plt.xlabel('weights')
# plt.ylabel('loss')
# plt.show()

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 첫 번째 서브플롯: loss_val_list 그래프
axs[0].plot(loss_val_list)
axs[0].set_xlabel('epochs')
axs[0].set_ylabel('loss')
axs[0].set_title('Loss over Epochs')

# 두 번째 서브플롯: w_val_list 그래프
axs[1].plot(w_val_list)
axs[1].set_xlabel('epochs')
axs[1].set_ylabel('weights')
axs[1].set_title('Weights over Epochs')

# 세 번째 서브플롯: w_val_list vs. loss_val_list 그래프
axs[2].scatter(w_val_list, loss_val_list)
axs[2].set_xlabel('weight')
axs[2].set_ylabel('loss')
axs[2].set_title('Loss vs. Weights')

plt.tight_layout()  # 서브플롯 간의 간격 조절
plt.show()
