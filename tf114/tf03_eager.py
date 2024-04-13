import tensorflow as tf
print(tf.__version__)   # 1.14.0
print(tf.executing_eagerly())   # False // 텐서2에서 1의 문법을 쓰기 위함


# 즉시실행모두 -> 텐서1의 그래프형태의 구성없이 
# 자연스러운 파이썬 문법으로 실행시킨다.

# 즉시실행모드 켠다.
# tf.compat.v1.disable_eager_execution()  # 즉시 실행모드 끈다. // 텐서플로 1.0 문법 // 디폴트
tf.compat.v1.enable_eager_execution()   # 즉시 실행모드 켠다. // 텐서플로 2.0 사용 가능

print(tf.executing_eagerly())   # True

hello = tf.constant('hello world')

sess = tf.compat.v1.Session() 
print(sess.run(hello))


'''
가상환경    즉시 실행 모드      사용가능
1.14.0      disable (디폴트)    가능 @@@@
1.14.0      enable              에러
2.9.0       disable             가능 @@@@
2.9.0       enable  (디폴트)     에러
'''

'''
텐서 1은 '그래프연산' 모드
텐서 2은 '즉시실행' 모드

tf.compat.v1.enable_eager_execution() # 즉시 실행모드 켜
                -> 텐서2의 디폴트

tf.compat.v1.disnable_eager_execution() # 즉시 실행모드 꺼
                -> 그래프 연산모드로 돌아간다
                -> 텐서1 코드를 쓸수 있다.

tf.executing_eagerly() # True면 즉시 실행모드, 텐서2 코드만 써야한다.
                        # False면 그래프 연산모드 -> 텐서1 코드를 쓸 수 있다.


'''
