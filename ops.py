import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib import image


@slim.add_arg_scope
def conv(inputs, num_filters, kernel_size, stride=1,
         dropout_rate=None, scope=None, outputs_collections=None, is_relu=True):
    with tf.variable_scope(scope, "xx", [inputs]) as sc:
        net = slim.conv2d(inputs=inputs, num_outputs=num_filters, kernel_size=kernel_size, stride=stride)
        net = slim.instance_norm(inputs=net)

        if is_relu:
            net = tf.nn.relu(net)

        if dropout_rate:
            net = tf.nn.dropout(net, keep_prob=dropout_rate)

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


@slim.add_arg_scope
def residual_block(inputs, num_filters, kernel_size, dropout_rate, scope=None, outputs_collections=None):
    with tf.variable_scope(scope, "residual_blockx", [inputs]) as sc:
        net = inputs

        net = conv(inputs=net, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate,
                   scope="conv_block_" + str(1))

        net = conv(inputs=net, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate,
                   scope="conv_block_" + str(2), is_relu=False)

        net = net + inputs

        net = tf.nn.relu(net)

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


@slim.add_arg_scope
def residual_block_down(inputs, num_filters, kernel_size, dropout_rate, scope=None, outputs_collections=None):
    with tf.variable_scope(scope, "residual_block_downx", [inputs]) as sc:
        net = conv(inputs=inputs, num_filters=num_filters, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate,
                   scope="strided_conv_block")

        net = residual_block(inputs=net, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate,
                             scope="residual_block_" + str(1))

        net = residual_block(inputs=net, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate,
                             scope="residual_block_" + str(2))

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


@slim.add_arg_scope
def residual_block_up(inputs, num_filters, kernel_size, dropout_rate, scope=None, outputs_collections=None):
    with tf.variable_scope(scope, "residual_block_upx", [inputs]) as sc:
        net = tf.image.resize_nearest_neighbor(images=inputs, size=[inputs.shape[1] * 2, inputs.shape[2] * 2])

        net = conv(inputs=net, num_filters=num_filters, kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate,
                   scope="conv_block")

        net = residual_block(inputs=net, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate,
                             scope="residual_block_" + str(1))

        net = residual_block(inputs=net, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate,
                             scope="residual_block_" + str(2))

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


def generator_network(input1, input2, dropout_rate=None, is_training=True, reuse=None, scope=None):
    with tf.variable_scope(scope, "generator", [input1, input2], reuse=reuse) as sc:
        end_points_collection = sc.name + "_end_points"

        with slim.arg_scope([slim.dropout], is_training=is_training), \
            slim.arg_scope([slim.conv2d, slim.conv2d_transpose, conv, residual_block, residual_block_down, residual_block_up],
                           outputs_collections=end_points_collection), \
            slim.arg_scope([conv, residual_block, residual_block_down, residual_block_up], dropout_rate=dropout_rate):

            net = tf.concat([input1, input2], axis=3)

            down_1 = residual_block_down(inputs=net, num_filters=64, kernel_size=3, scope="residual_block_down_" + str(1)) # 128

            down_2 = residual_block_down(inputs=down_1, num_filters=128, kernel_size=3, scope="residual_block_down_" + str(2)) # 64

            down_3 = residual_block_down(inputs=down_2, num_filters=256, kernel_size=3, scope="residual_block_down_" + str(3)) # 32

            down_4 = residual_block_down(inputs=down_3, num_filters=512, kernel_size=3, scope="residual_block_down_" + str(4)) # 16

            down_5 = residual_block_down(inputs=down_4, num_filters=512, kernel_size=3, scope="residual_block_down_" + str(5)) # 8

            down_6 = residual_block_down(inputs=down_5, num_filters=512, kernel_size=3, scope="residual_block_down_" + str(6)) # 4

            up_1 = residual_block_up(inputs=down_6, num_filters=512, kernel_size=3, scope="residual_block_up_" + str(1)) # 8

            up_2 = residual_block_up(inputs=tf.concat([down_5, up_1], axis=-1), num_filters=512, kernel_size=3, scope="residual_block_up_" + str(2)) # 16

            up_3 = residual_block_up(inputs=tf.concat([down_4, up_2], axis=-1), num_filters=256, kernel_size=3, scope="residual_block_up_" + str(3)) # 32

            up_4 = residual_block_up(inputs=tf.concat([down_3, up_3], axis=-1), num_filters=128, kernel_size=3, scope="residual_block_up_" + str(4)) # 64

            up_5 = residual_block_up(inputs=tf.concat([down_2, up_4], axis=-1), num_filters=64, kernel_size=3, scope="residual_block_up_" + str(5)) # 128

            up_6 = residual_block_up(inputs=tf.concat([down_1, up_5], axis=-1), num_filters=3, kernel_size=3, scope="residual_block_up_" + str(6)) # 256

            change = conv(inputs=up_6, num_filters=3, kernel_size=7, stride=1, scope="conv_change", is_relu=False)  # 256 # Conv cuối phải bỏ relu trước khi qua tanh

            change = tf.nn.tanh(change)  # 256

            alpha = conv(inputs=up_6, num_filters=3, kernel_size=7, stride=1, scope="conv_alpha", is_relu=False)  # 256 # Conv cuối phải bỏ relu trước khi qua tanh

            alpha = tf.nn.sigmoid(alpha)  # 256

            out = (tf.ones_like(alpha) - alpha) * change + alpha * input1  # 256

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        return out, end_points


def generator_arg_scope(weight_decay=1e-4, instance_norm_epsilon=1.1e-5):
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(scale=weight_decay),
                        activation_fn=None,
                        biases_initializer=tf.zeros_initializer(),
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
        with slim.arg_scope([slim.instance_norm],
                            scale=True,
                            epsilon=instance_norm_epsilon) as scope:
            return scope


#with slim.arg_scope(generator_arg_scope()):
#    generator = generator_network
#
#inputs = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 6])
#feed_inputs = np.random.rand(1, 256, 256, 6)
#output, _ = generator(inputs=inputs)
#
#initializer = tf.global_variables_initializer()
#
#with tf.Session() as sess:
#    sess.run(initializer)
#
#    out = sess.run(output, feed_dict={inputs: feed_inputs})
#
#    print(out, out.shape)
#
#from tensorflow.contrib.slim.nets import vgg
