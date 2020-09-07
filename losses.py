import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib import slim


def l1_loss(source, predict):
    subtract = source - predict
    loss = tf.reduce_mean(tf.abs(subtract))
    return loss


def l2_loss(source, predict):
    subtract = source - predict
    loss = tf.reduce_mean(tf.square(subtract))
    return loss


def wing_loss(source, predict, w=1.0, epsilon=0.1):
    subtract = tf.abs(source - predict)
    loss = tf.where(tf.less(subtract, w * tf.ones_like(subtract)), w * tf.math.log(1 + subtract / epsilon), subtract - (w - w * tf.math.log(1 + w / epsilon)))
    loss = tf.reduce_mean(loss)
    return loss
