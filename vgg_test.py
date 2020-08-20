import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import vgg

from image_translation import ImageTranslation

inputs = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3])

with slim.arg_scope(vgg.vgg_arg_scope()):
    net, end_points = vgg.vgg_19(inputs, is_training=False, spatial_squeeze=False)

print("net: {}".format(net))
print("end_points: {}".format(end_points))
