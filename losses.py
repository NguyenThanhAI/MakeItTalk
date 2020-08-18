import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib import slim


def l1_loss(source, predict):
    subtract = source - predict
    loss = tf.reduce_mean(tf.abs(subtract))
    return loss

