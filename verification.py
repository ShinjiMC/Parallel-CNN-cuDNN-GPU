import tensorflow as tf

# Verifica si TensorFlow detecta alguna GPU
if tf.test.gpu_device_name():
    print("GPU encontrada:")
    print(tf.test.gpu_device_name())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
else:
    print("No se encontró una GPU. TensorFlow está utilizando la CPU.")