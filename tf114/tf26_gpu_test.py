import tensorflow as tf
# T1 -> 그래프 연산
# T2 -> 즉시 실행 모드

# tf.compat.v1.enable_eager_execution()   # 즉시 실행 모드 켜
# 텐서플로 버젼 :  1.14.0
# 즉시 실행 모드 :  True
# gpu 없다...

# 텐서플로 버젼 :  2.9.0
# 즉시 실행 모드 :  True
# PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')

# tf.compat.v1.disable_eager_execution()    # 즉시 모드 꺼 -> 그래프 연산모드 -> T1 코드를 쓸수 있다.
# 텐서플로 버젼 :  1.14.0
# 즉시 실행 모드 :  False
# gpu 없다...

# 텐서플로 버젼 :  2.9.0
# 즉시 실행 모드 :  False
# PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')

print("텐서플로 버젼 : ", tf.__version__)
print("즉시 실행 모드 : ", tf.executing_eagerly())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try : 
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(gpus[0])
    except RuntimeError as e:
        print(e)
else : 
    print("gpu 없다...")