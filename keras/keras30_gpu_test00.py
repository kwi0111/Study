import tensorflow as tf
print(tf.__version__)   # 2.15.0


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus):
    print("쥐피유 돈다..")
else:
    print("지피유 안돈다..")
    
    
    
    