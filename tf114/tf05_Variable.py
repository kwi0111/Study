import tensorflow as tf
sess = tf.compat.v1.Session()

a = tf.Variable([2], dtype=tf.float32)
b = tf.Variable([3], dtype=tf.float32) 

# 텐서플로우에서 변수를 정의할때 변수 초기화를 해야한다.
init =tf.compat.v1.global_variables_initializer()   #  모든 변수 초기화
sess.run(init)  # 세션을 통해 변수 초기화 연산

print(sess.run(a + b))




