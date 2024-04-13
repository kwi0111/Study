import tensorflow as tf
import numpy as np

# TensorFlow 1.x에서 GPU 설정
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # GPU 메모리 점진적 할당 활성화
sess = tf.compat.v1.Session(config=config)

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(777)

# GPU 사용 가능 여부 확인
if tf.test.is_gpu_available():
    print("GPU is 사용 가능하다")
else:
    print("GPU is 사용 불가능하다")

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(777)

#1. 데이터

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

#2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, [None,28,28,1])    # input_shape
y = tf.compat.v1.placeholder(tf.float32, [None,10]) 

# Layer1
w1 = tf.compat.v1.get_variable('w1', shape=[2,2,1,64], initializer=tf.contrib.layers.xavier_initializer())  # 커널사이즈 (2,2) / 컬러(채널 1개, 흑백) / 128개 필터(아웃풋) /
b1 = tf.compat.v1.Variable(tf.zeros([64]), name='b1')   # bias는 필터의 갯수와 동일하다.

L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID')    # 필터가 이미지를 한 픽셀씩 이동하며 적용
L1 = L1 + b1    # L1 += b1
L1 = tf.nn.relu(L1)
# L1 = tf.nn.dropout(L1, keep_prob=0.7)   # 70%를 살려내겠다.
L1 = tf.nn.dropout(L1, rate=0.3)   # 70%를 살려내겠다 // model.add(Dropout(0.3))
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
# model.add(Conv2d(64, kenel_size(2,2), stride=(1,1), input_shape=(28,28,1)))
print(L1)   # Tensor("Relu:0", shape=(?, 27, 27, 128), dtype=float32)
print(L1_maxpool)   # Tensor("MaxPool2d:0", shape=(?, 13, 13, 128), dtype=float32)

# Layer2
w2 = tf.compat.v1.get_variable('w2', shape=[3,3,64,32], initializer=tf.contrib.layers.xavier_initializer()) 
b2 = tf.compat.v1.Variable(tf.zeros([32]), name='b2')   # bias는 필터의 갯수와 동일하다.

L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='SAME')    # 필터가 이미지를 한 픽셀씩 이동하며 적용
L2 = L2 + b2    # L1 += b1
L2 = tf.nn.selu(L2)
L2 = tf.nn.dropout(L2, keep_prob=0.9)   # 90%를 살려내겠다 // model.add(Dropout(0.1))
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
# model.add(Conv2d(64, kenel_size(2,2), stride=(1,1), input_shape=(28,28,1)))
print(L2)   # Tensor("dropout_1/mul_1:0", shape=(?, 13, 13, 64), dtype=float32)
print(L2_maxpool)   # Tensor("MaxPool2d_1:0", shape=(?, 6, 6, 64), dtype=float32)

# Layer3
w3 = tf.compat.v1.get_variable('w3', shape=[3,3,32,16], initializer=tf.contrib.layers.xavier_initializer()) 
b3 = tf.compat.v1.Variable(tf.zeros([16]), name='b3')   # bias는 필터의 갯수와 동일하다.

L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME')    # 필터가 이미지를 한 픽셀씩 이동하며 적용
L3 = L3 + b3    # L1 += b1
L3 = tf.nn.elu(L3)
print(L3)   # Tensor("Elu:0", shape=(?, 6, 6, 32), dtype=float32)

# Flatten
L_flat = tf.reshape(L3, [-1, 6*6*16])
print("플래튼 : ", L_flat)  # 플래튼 :  Tensor("Reshape:0", shape=(?, 1152), dtype=float32)

# Layer4 DNN
w4 = tf.compat.v1.get_variable('w4', shape=[6*6*16, 32], initializer=tf.contrib.layers.xavier_initializer()) 
b4 = tf.compat.v1.Variable(tf.zeros([32]), name='b4')
L4 = tf.nn.relu(tf.matmul(L_flat, w4) + b4)
L4 = tf.nn.dropout(L4, rate=0.3)

# Layer5 DNN
w5 = tf.compat.v1.get_variable('w5', shape=[32, 10], initializer=tf.contrib.layers.xavier_initializer()) 
b5 = tf.compat.v1.Variable(tf.zeros([10]), name='b5')
L5 = tf.nn.relu(tf.matmul(L4, w5) + b5)
hypothesis = tf.nn.softmax(L5)

#3-1. 컴파일
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=hypothesis))
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis + 1e-7 ),axis=1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y),dtype=tf.float32))

batch_size = 64
total_batch = int(len(x_train) / batch_size)

epochs = 101
for step in range(epochs):
    avg_loss = 0    # loss == cost
    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size
        
        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        
        loss_val, _ = sess.run([loss,train], feed_dict=feed_dict)
        
        avg_loss += loss_val / total_batch
        
        
    if step %1 == 0:
        print(step, "loss : ", avg_loss)

y_predict = sess.run(hypothesis, feed_dict = {x:x_test})
# print(y_predict)   
y_predict = sess.run(tf.argmax(y_predict, 1))
# print(y_predict) 
y_data_arg = np.argmax(y_test, 1)
# print(y_data_arg)    


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict, y_data_arg)
print('ACC : ', acc) 
# ACC :  0.9863
sess.close()




