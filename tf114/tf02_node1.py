# 텐서플로우는 그래프 연산 방식 -> 노드를 정리해줘야함.
# 1 + 2 3 // 총 4개 준비 되어야함.
import tensorflow as tf

# 3 + 4 = ?
node1 = tf.constant(3.0, tf.float32)    # 부동소수점의 데이터다.
node2 = tf.constant(4.0)
# node3 = node1 + node2
node3 = tf.add(node1, node2)

print(node3)    
# Tensor("Add:0", shape=(), dtype=float32)

sess = tf.Session()
print(sess.run(node3))  # 7.0
