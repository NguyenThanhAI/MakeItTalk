import os
import time
from datetime import datetime
import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim import nets

from ops import generator_arg_scope, generator_network, discriminator_arg_scope, discriminator_network
from read_tfrecord import get_batch
from losses import l1_loss


class ImageTranslationConfig(object):
    def __init__(self, input_size=256, learning_rate=5e-4, learning_rate_decay_type="exponential_decay",
                 decay_steps=10000, decay_rate=0.9, momentum=0.9, lambda_a=1., lambda_c=1., lambda_g=100.,
                 use_cycle_loss=False, use_discriminator=True,
                 weight_decay=1e-6, is_loadmodel=False,
                 per_process_gpu_memory_fraction=1.0,
                 summary_dir="./summary", model_dir="./saved_model",
                 vgg_model_dir="/thanhnc/vgg_19", vgg_checkpoint="vgg_19.ckpt", generator_checkpoint_name=None,
                 discriminator_checkpoint_name=None, discriminator_update_steps=4, dis_gen_learning_rate_ratio=2.,
                 dataset_path="/thanhnc/VoxCeleb2/data/tfrecord/training.tfrecord", batch_size=8, num_epochs=100,
                 summary_frequency=10, save_network_frequency=10000,
                 is_training=True, optimizer="adam"):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.learning_rate_decay_type = learning_rate_decay_type
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.momentum = momentum
        self.lambda_a = lambda_a
        self.lambda_c = lambda_c
        self.lambda_g = lambda_g
        self.use_cycle_loss = use_cycle_loss
        self.use_discriminator = use_discriminator
        self.weight_decay = weight_decay
        self.is_loadmodel = is_loadmodel
        self.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        self.summary_dir = summary_dir
        self.model_dir = model_dir
        self.vgg_model_dir = vgg_model_dir
        self.vgg_checkpoint = vgg_checkpoint
        self.generator_checkpoint_name = generator_checkpoint_name
        self.discriminator_checkpoint_name = discriminator_checkpoint_name
        self.discriminator_update_steps = discriminator_update_steps
        self.dis_gen_learning_rate_ratio = dis_gen_learning_rate_ratio
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.summary_frequency = summary_frequency
        self.save_network_frequency = save_network_frequency
        self.is_training = is_training
        self.optimizer = optimizer


class ImageTranslation(object):
    def  __init__(self, config: ImageTranslationConfig):
        self.config = config
        self.source_image, self.target_image, self.source_landmarks, self.target_landmarks, self.epoch_now = get_batch(tfrecord_path=self.config.dataset_path,
                                                                                                                       batch_size=self.config.batch_size,
                                                                                                                       num_epochs=self.config.num_epochs)

        with slim.arg_scope(generator_arg_scope()):
            self.predicted_target_image, _ = generator_network(inputs=tf.concat([self.source_image, self.target_landmarks], axis=3),
                                                               is_training=self.config.is_training,
                                                               reuse=False, scope="image_translation_module")
            if self.config.use_cycle_loss:
                self.predicted_source_image, _ = generator_network(inputs=tf.concat([self.predicted_target_image, self.source_landmarks], axis=3),
                                                                   is_training=self.config.is_training,
                                                                   reuse=True, scope="image_translation_module")
        self.perceptual_target_image = tf.concat([self.target_image, self.predicted_target_image], axis=0)
        if self.config.use_cycle_loss:
            self.perceptual_image = tf.concat([self.target_image, self.predicted_target_image, self.source_image, self.source_landmarks], axis=0)
        with slim.arg_scope(nets.vgg.vgg_arg_scope()):
            if not self.config.use_cycle_loss:
                self.perceptual_pred_target_image = nets.vgg.vgg_19(inputs=self.perceptual_target_image, is_training=False, spatial_squeeze=False)[1]["vgg_19/conv5/conv5_4"]
            else:
                self.perceptual_pred_image = nets.vgg.vgg_19(inputs=self.perceptual_image, is_training=False, spatial_squeeze=False)[1]["vgg_19/conv5/conv5_4"]
        if not self.config.use_cycle_loss:
            self.output_perceptual_target_image, self.output_perceptual_predicted_target_image = tf.split(value=self.perceptual_pred_target_image,
                                                                                                          num_or_size_splits=2,
                                                                                                          axis=0)
        else:
            self.output_perceptual_target_image, self.output_perceptual_predicted_target_image, self.output_perceptual_source_image, self.output_perceptual_predicted_source_image = \
            tf.split(value=self.perceptual_pred_image,
                     num_or_size_splits=4,
                     axis=0)

        if self.config.use_discriminator:
            if self.config.use_cycle_loss:
                self.fake_inputs = tf.concat([self.predicted_target_image, self.predicted_source_image], axis=0)
                self.real_inputs = tf.concat([self.target_image, self.source_image], axis=0)
            else:
                self.fake_inputs = self.predicted_target_image
                self.real_inputs = self.target_image

            with slim.arg_scope(discriminator_arg_scope()):
                self.fake_outputs, _ = discriminator_network(inputs=self.fake_inputs, is_training=True,
                                                             reuse=False, scope="discriminator")
                self.real_outputs, _ = discriminator_network(inputs=self.real_inputs, is_training=True,
                                                             reuse=True, scope="discriminator")

        self.global_step = tf.train.get_or_create_global_step()
        self.global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + \
            tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
        #print("self.global_variables: {}".format(self.global_variables))
        self.generator_trainable_variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                               scope="image_translation_module")
        #print("self.generator_trainable_variables: {}".format(self.generator_trainable_variables))
        self.generator_saver = tf.train.Saver(var_list=self.generator_trainable_variables + [self.global_step])
        if self.config.use_discriminator:
            self.discriminator_trainable_variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                       scope="discriminator")
            self.discriminator_saver = tf.train.Saver(var_list=self.discriminator_trainable_variables)
        self.vgg_variables = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_19")
        #print("self.vgg_variables: {}".format(self.vgg_variables))
        self.vgg_saver = tf.train.Saver(var_list=self.vgg_variables)

        with tf.name_scope("loss"):
            with tf.name_scope("generator"):
                self.generator_regularization_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in self.generator_trainable_variables])
                tf.summary.scalar("regularization loss", self.generator_regularization_loss)
                self.reconstruction_loss = l1_loss(source=self.target_image, predict=self.predicted_target_image)
                tf.summary.scalar("reconstruction loss", self.reconstruction_loss)
                if not self.config.use_cycle_loss:
                    self.perceptual_loss = l1_loss(source=self.output_perceptual_target_image,
                                                   predict=self.output_perceptual_predicted_target_image)
                    tf.summary.scalar("perceptual loss", self.perceptual_loss)
                    if not self.config.use_discriminator:
                        self.generator_loss = self.reconstruction_loss + self.config.lambda_a * self.perceptual_loss + self.config.weight_decay * self.generator_regularization_loss
                    else:
                        self.gen_adv_loss = tf.reduce_mean(0.5 * tf.square(self.fake_outputs - tf.ones_like(self.fake_outputs)))
                        tf.summary.scalar("generator adversarial loss", self.gen_adv_loss)
                        self.generator_loss = self.reconstruction_loss + self.config.lambda_a * self.perceptual_loss + self.config.lambda_g * self.gen_adv_loss + self.config.weight_decay * self.generator_regularization_loss
                    tf.summary.scalar("total loss", self.generator_loss)
                else:
                    self.perceptual_loss = l1_loss(source=self.output_perceptual_target_image,
                                                   predict=self.output_perceptual_predicted_target_image) + \
                                           l1_loss(source=self.output_perceptual_source_image,
                                                   predict=self.output_perceptual_predicted_source_image)
                    tf.summary.scalar("perceptual loss", self.perceptual_loss)
                    self.cycle_loss = l1_loss(source=self.source_image, predict=self.predicted_source_image)
                    tf.summary.scalar("cycle loss", self.cycle_loss)
                    if not self.config.use_discriminator:
                        self.generator_loss = self.reconstruction_loss + self.config.lambda_c * self.cycle_loss + self.config.lambda_a * self.perceptual_loss + self.config.weight_decay * self.generator_regularization_loss
                    else:
                        self.gen_adv_loss = tf.reduce_mean(0.5 * tf.square(self.fake_outputs - tf.ones_like(self.fake_outputs)))
                        tf.summary.scalar("generator adversarial loss", self.gen_adv_loss)
                        self.generator_loss = self.reconstruction_loss + self.config.lambda_c * self.cycle_loss + self.config.lambda_a * self.perceptual_loss + self.config.lambda_g * self.gen_adv_loss + self.config.weight_decay * self.generator_regularization_loss
                    tf.summary.scalar("total loss", self.generator_loss)

            if self.config.use_discriminator:
                with tf.name_scope("discriminator"):
                    self.discriminator_regularization_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in self.discriminator_trainable_variables])
                    tf.summary.scalar("discriminator regularization loss", self.discriminator_regularization_loss)
                    self.dis_adv_loss = tf.reduce_mean(0.5 * tf.square(self.fake_outputs)) + tf.reduce_mean(0.5 * tf.square(self.real_outputs - tf.ones_like(self.real_outputs)))
                    tf.summary.scalar("discriminator adversarial loss", self.dis_adv_loss)
                    self.discriminator_loss = self.dis_adv_loss + self.config.weight_decay * self.discriminator_regularization_loss
                    tf.summary.scalar("discriminator loss", self.discriminator_loss)

        with tf.name_scope("image"):
            tf.summary.image("source image", tf.cast(self.source_image, dtype=tf.uint8))
            tf.summary.image("target image", tf.cast(self.target_image, dtype=tf.uint8))
            if self.config.use_cycle_loss:
                tf.summary.image("predicted source image", tf.cast(self.predicted_source_image, dtype=tf.uint8))
            tf.summary.image("predicted target image", tf.cast(self.predicted_target_image, tf.uint8))

        with tf.name_scope("optimizer"):
            if self.config.learning_rate_decay_type == "constant":
                self.learning_rate = self.config.learning_rate
            elif self.config.learning_rate_decay_type == "piecewise_constant":
                self.learning_rate = tf.train.piecewise_constant(x=self.global_step,
                                                                 boundaries=[20000, 200000, 500000],
                                                                 values=[5e-4, 2.5e-4, 1e-4, 5e-5])
            elif self.config.learning_rate_decay_type == "exponential_decay":
                self.learning_rate = tf.train.exponential_decay(learning_rate=self.config.learning_rate,
                                                                global_step=self.global_step,
                                                                decay_steps=self.config.decay_steps,
                                                                decay_rate=self.config.decay_rate,
                                                                staircase=True)
            elif self.config.learning_rate_decay_type == "linear_cosine_decay":
                self.learning_rate = tf.train.linear_cosine_decay(learning_rate=self.config.learning_rate,
                                                                  global_step=self.global_step,
                                                                  decay_steps=self.config.decay_steps)
            tf.summary.scalar("learning rate", self.learning_rate)
            if self.config.optimizer.lower() == "adam":
                self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
                if self.config.use_discriminator:
                    self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.config.dis_gen_learning_rate_ratio * self.config.learning_rate)
            elif self.config.optimizer.lower() == "rms" and self.config.optimizer.lower() == "rmsprop":
                self.generator_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                if self.config.use_discriminator:
                    self.discriminator_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.dis_gen_learning_rate_ratio * self.learning_rate)
            elif self.config.optimizer.lower() == "momentum":
                self.generator_optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                                      momentum=self.config.momentum)
                if self.config.use_discriminator:
                    self.discriminator_optimizer = tf.train.MomentumOptimizer(learning_rate=self.config.dis_gen_learning_rate_ratio * self.learning_rate,
                                                                              momentum=self.config.momentum)
            elif self.config.optimizer.lower() == "grad_descent":
                self.generator_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
                if self.config.use_discriminator:
                    self.discriminator_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.dis_gen_learning_rate_ratio * self.config.learning_rate)
            else:
                raise ValueError("Optimizer {} was not recognized".format(self.config.optimizer))

        self.generator_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="image_translation_module")
        with tf.control_dependencies(self.generator_update_ops):
            self.generator_train_op = self.generator_optimizer.minimize(loss=self.generator_loss, global_step=self.global_step, var_list=self.generator_trainable_variables)

        if self.config.use_discriminator:
            self.discriminator_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="discriminator")
            with tf.control_dependencies(self.discriminator_update_ops):
                self.discriminator_train_op = self.discriminator_optimizer.minimize(loss=self.discriminator_loss, var_list=self.discriminator_trainable_variables)

        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.config.per_process_gpu_memory_fraction)
        self.config_gpu = tf.ConfigProto(gpu_options=self.gpu_options)
        self.sess = tf.Session(config=self.config_gpu)

        self.merged = tf.summary.merge_all()

        if self.config.is_loadmodel:
            self.writer = tf.summary.FileWriter(logdir=self.config.summary_dir)
        else:
            self.writer = tf.summary.FileWriter(logdir=self.config.summary_dir, graph=self.sess.graph)

        self.restore_or_initialize_network(generator_checkpoint_name=self.config.generator_checkpoint_name,
                                           discriminator_checkpoint_name=self.config.discriminator_checkpoint_name)

    def restore_or_initialize_network(self, generator_checkpoint_name=None, discriminator_checkpoint_name=None):
        self.sess.run(tf.global_variables_initializer())
        if self.config.is_loadmodel:
            if generator_checkpoint_name is not None:
                self.generator_saver.restore(sess=self.sess, save_path=os.path.join(self.config.model_dir, generator_checkpoint_name))
            else:
                self.generator_saver.restore(sess=self.sess, save_path=tf.train.latest_checkpoint(checkpoint_dir=self.config.model_dir))
            if self.config.use_discriminator:
                if discriminator_checkpoint_name is not None:
                    self.discriminator_saver.restore(sess=self.sess, save_path=os.path.join(self.config.model_dir, discriminator_checkpoint_name))
                else:
                    self.discriminator_saver.restore(sess=self.sess, save_path=tf.train.latest_checkpoint(checkpoint_dir=self.config.model_dir))
            print("Successfully load model at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        else:
            print("Successfully initialize model at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        self.vgg_saver.restore(sess=self.sess, save_path=os.path.join(self.config.vgg_model_dir, self.config.vgg_checkpoint))
        print("Successfully restore VGG 19 weights at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

    def get_global_step(self):
        return tf.train.global_step(sess=self.sess, global_step_tensor=self.global_step)

    def save_network(self, global_step):
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir, exist_ok=True)
        generator_checkpoint_name = "Generator-" + str(global_step).zfill(7)
        self.generator_saver.save(sess=self.sess, save_path=os.path.join(self.config.model_dir, generator_checkpoint_name))
        print("Save generator network at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        if self.config.use_discriminator:
            discriminator_checkpoint_name = "Discriminator-" + str(global_step).zfill(7)
            self.discriminator_saver.save(sess=self.sess, save_path=os.path.join(self.config.model_dir, discriminator_checkpoint_name))
            print("Save discriminator network at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

    def add_summary(self, global_step):

        summary = self.sess.run(self.merged)

        self.writer.add_summary(summary=summary, global_step=global_step)
        print("Add summary at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

    def train(self):
        try:
            while True:
                start = time.time()
                _, gen_cost, epoch = self.sess.run([self.generator_train_op, self.generator_loss, self.epoch_now])
                global_step = self.get_global_step()
                if self.config.use_discriminator:
                    if global_step % self.config.discriminator_update_steps == 0:
                        _, dis_cost = self.sess.run([self.discriminator_train_op, self.discriminator_loss])
                        print("Discriminator loss: {},".format(np.round(dis_cost, 3)), end=" ")
                if global_step % self.config.summary_frequency == 0:
                    self.add_summary(global_step=global_step)
                if global_step % self.config.save_network_frequency == 0:
                    self.save_network(global_step=global_step)
                end = time.time()
                print("Step: {}, generator loss: {}, epoch: {}, takes time: {}".format(global_step, np.round(gen_cost, 3), epoch, np.round(end - start, 3)))
        except tf.errors.OutOfRangeError:
            global_step = self.get_global_step()
            self.save_network(global_step=global_step)
            print("Training process finished")
            pass
