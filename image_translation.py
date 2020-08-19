import os
import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim

from ops import generator_arg_scope, generator_network
from read_tfrecord import get_batch


class ImageTranslationConfig(object):
    def __init__(self, input_size, learning_rate, learning_rate_decay_type, use_cycle_loss,
                 vgg_checkpoint, checkpoint_name, dataset_path, batch_size, num_epochs):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.learning_rate_decay_type = learning_rate_decay_type
        self.use_cycle_loss = use_cycle_loss
        self.vgg_checkpoint = vgg_checkpoint
        self.checkpoint_name = checkpoint_name
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs


class ImageTranslation(object):
    def  __init__(self, config: ImageTranslationConfig):
        self.config = config
        self.source_image, self.target_image, self.source_landmarks, self.target_landmarks = get_batch(tfrecord_path=self.config.dataset_path,
                                                                                                       batch_size=self.config.batch_size,
                                                                                                       num_epochs=self.config.num_epochs)

        with slim.arg_scope(generator_arg_scope()):
            self.predicted_target_image, _ = generator_network()