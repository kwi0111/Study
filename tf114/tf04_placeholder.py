import tensorflow as tf
print(tf.__version__)  
print(tf.executing_eagerly()) 

tf.compat.v1.disable_eager_execution() 

node1 = tf.constant(30.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)    # 인풋값을 넣는 값
add_node = a + b

print(sess.run(add_node, feed_dict={a:3, b:4}))
print(sess.run(add_node, feed_dict={a:30, b:4.5}))

add_and_triple = add_node * 3   # 그래프가 다시 만들어졌다.
print(add_and_triple)   # Tensor("mul:0", dtype=float32)
print(sess.run(add_and_triple, feed_dict={a:3, b:4})) # a와 b를 모르니까 다시 명시를 해줘야한다.
