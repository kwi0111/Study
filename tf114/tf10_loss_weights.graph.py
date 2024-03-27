import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

x = [1,2,3]
y = [1,2,3]

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
        
print('==================================== W history ==========================================')
print(w_history)
print('=================================== LOSS history ==========================================')
print(loss_history)


plt.plot(w_history, loss_history)
plt.xlabel('weights')
plt.ylabel('loss')
plt.show()



