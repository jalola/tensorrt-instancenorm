import os
import tensorflow as tf
import tensorflow_addons as tfa
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[int(0)], True)

tf.keras.backend.set_learning_phase(0)

model = tf.keras.models.Sequential([
  tf.keras.layers.Reshape((512,512,1), input_shape=(512, 512)),
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), data_format="channels_last", padding="SAME"),

  tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform"),
  tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), data_format="channels_last", padding="SAME")
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
              
output = "tf_model/saved_model_1.0/"
onnx_file = "onnx/keras_instance_norm_1.onnx"

tf.saved_model.save(model, str(output))

# I use this command to convert tensorflow model to onnx
!python3.6 -m tf2onnx.convert --opset 11  --saved-model $output --output $onnx_file
