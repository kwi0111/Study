import tensorflow as tf
print(tf.__version__)  
print(tf.executing_eagerly()) 

tf.compat.v1.disable_eager_execution() # TensorFlow의 즉시 실행 모드를 비활성화. TensorFlow 1.x에서 사용되던 세션 기반 실행 방식으로 돌아가기 위함

node1 = tf.constant(30.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32)    # 입력값을 받기 위한 placeholder를 정의
b = tf.compat.v1.placeholder(tf.float32)    # 인풋값을 넣는 값
add_node = a + b

# 세션을 사용하여 연산 실행
print(sess.run(add_node, feed_dict={a:3, b:4})) # placeholder, feed_dict 짝이다.
print(sess.run(add_node, feed_dict={a:30, b:4.5}))

# 추가 연산 정의 및 실행
add_and_triple = add_node * 3   # 그래프가 다시 만들어졌다.
print(add_and_triple)   # Tensor("mul:0", dtype=float32)
print(sess.run(add_and_triple, feed_dict={a:3, b:4})) # a와 b를 모르니까 다시 명시를 해줘야한다.
