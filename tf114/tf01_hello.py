import tensorflow as tf
print(tf.__version__)   # 1.14.0


print("hello world")


hello = tf.constant('hello world')     # 상수 
print(hello)    # Tensor("Const:0", shape=(), dtype=string) // 출력을 지정해주지 않으면 과정만 출력

sess = tf.Session() # 정의를 해줘야한다. 텐서1에서는 무조건 거쳐야한다.
print(sess.run(hello))

