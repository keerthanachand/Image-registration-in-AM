import tensorflow as tf
import tensorrt

print("TF version: ",tf.__version__)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  print("Name:", gpu.name, "  Type:", gpu.device_type)