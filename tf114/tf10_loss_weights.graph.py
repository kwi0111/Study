import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

# x = [1,2,3]
# y = [1,2,3]

# 실습
x = [1,2]
y = [1,2]



w = tf.compat.v1.placeholder(tf.float32)    # feed_dict 

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))    # 예측된 거에서 y를 빼서 제곱한거의 평균 MSE
w_history = []
loss_history = []

with tf.compat.v1.Session() as sess :
    for i in range(-30, 50):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict = {w : curr_w})
        
        w_history.append(curr_w)
        loss_history.append(curr_loss)
        
# print('==================================== W history ==========================================')
# print(w_history)
# print('=================================== LOSS history ==========================================')
# print(loss_history)


# plt.plot(w_history, loss_history)
# plt.xlabel('weights')
# plt.ylabel('loss')
# plt.show()


# w가 30일 때의 인덱스를 찾습니다.
w_index = w_history.index(49)

# 해당 인덱스에 대한 손실 값을 확인합니다.
loss_at_w_30 = loss_history[w_index]

print("w가 30일 때의 손실:", loss_at_w_30)
