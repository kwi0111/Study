import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

# 실습
# 덧셈 : node3
# 뺄셈 : node4
# 곱셈 : node5
# 나눗셈 : node6
sess = tf.Session()
node3 = tf.add(node1, node2)
print(sess.run(node3))  # 덧셈 5.0

node4 = tf.subtract(node1, node2)
print(sess.run(node4))  # 뺄셈 -1.0

node5 = tf.multiply(node1, node2)
print(sess.run(node5))  # 곱셈 6.0

node6 = tf.divide(node1, node2)
print(sess.run(node6))  # 나눗셈 0.6666667
