import tensorflow as tf
tf.compat.v1.set_random_seed(777)



변수 = tf.compat.v1.Variable(tf.random_normal([2]), name='weight')
print(변수) # <tf.Variable 'weight:0' shape=(2,) dtype=float32_ref>

# 초기화 첫번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())   # 초기화 안하면 에러

aaa = sess.run(변수)
print('aaa : ', aaa)    # aaa :  [ 2.2086694  -0.73225045]
sess.close()

# 초기화 두번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

bbb = 변수.eval(session = sess) # 텐서플로 데이터형은 '변수'를 파이썬에서 볼 수 있게 바꿔준다.
print('bbb : ', bbb)    # bbb : bbb :  [ 2.2086694  -0.73225045]  // 런으로 안하고 .eval로 할수 있따
sess.close()

# 초기화 세번째
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval(session = sess) # InteractiveSession -> .eval로하면 편하다
print('ccc : ', ccc)        # ccc :  [ 2.2086694  -0.73225045]
sess.close()




