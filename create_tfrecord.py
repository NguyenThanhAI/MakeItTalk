import sys
import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import tensorflow as tf

def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    if  not isinstance(value, list):
        value = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _image_to_tfexample(source_image_data, target_image_data, source_landmarks_data, target_landmarks_data):
    return tf.train.Example(features=tf.train.Features(feature={"source_image": _bytes_feature(source_image_data),
                                                                "target_image": _bytes_feature(target_image_data),
                                                                "source_landmarks": _bytes_feature(source_landmarks_data),
                                                                "target_landmarks": _bytes_feature(target_landmarks_data)}))

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, default=r"K:\VoxCeleb2\data")
    parser.add_argument("--saved_dir", type=str, default=r"K:\VoxCeleb2\tfrecord")
    parser.add_argument("--split_factor", type=float, default=0.9)
    parser.add_argument("--per_process_gpu_memory_fraction", type=float, default=0.1)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_args()

    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)

    directory_list = []
    for root, dirs, files in os.walk(args.image_dir):
        if not dirs:
            directory_list.append(root)

    print("directory_list: {}".format(directory_list))
    np.random.shuffle(directory_list)

    training_directory_list = directory_list[:int(args.split_factor*len(directory_list))]
    validation_directory_list = directory_list[int(args.split_factor*len(directory_list)):]

    source_image_phl = tf.placeholder(dtype=tf.uint8, shape=[None, None, None])
    target_image_phl = tf.placeholder(dtype=tf.uint8, shape=[None, None, None])
    source_landmarks_phl = tf.placeholder(dtype=tf.uint8, shape=[None, None, None])
    target_landmarks_phl = tf.placeholder(dtype=tf.uint8, shape=[None, None, None])

    source_image_encoded = tf.image.encode_jpeg(source_image_phl, quality=99)
    target_image_encoded = tf.image.encode_jpeg(target_image_phl, quality=99)
    source_landmarks_encoded = tf.image.encode_jpeg(source_landmarks_phl, quality=99)
    target_landmarks_encoded = tf.image.encode_jpeg(target_landmarks_phl, quality=99)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.per_process_gpu_memory_fraction)
    config_gpu = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config_gpu) as sess:

        with tf.python_io.TFRecordWriter(os.path.join(args.saved_dir, "training.tfrecord")) as training_tfrecord_writer:
            for directory in tqdm(training_directory_list):
                images = os.listdir(directory)
                if len(images) != 4:
                    continue

                source_image = cv2.imread(os.path.join(directory, "source_image.jpg"))[:, :, ::-1]
                target_image = cv2.imread(os.path.join(directory, "target_image.jpg"))[:, :, ::-1]
                source_landmarks = cv2.imread(os.path.join(directory, "source_landmarks.jpg"))[:, :, ::-1]
                target_landmarks = cv2.imread(os.path.join(directory, "target_landmarks.jpg"))[:, :, ::-1]

                source_image_string, target_image_string, source_landmarks_string, target_landmarks_string = \
                sess.run([source_image_encoded, target_image_encoded, source_landmarks_encoded, target_landmarks_encoded],
                         feed_dict={source_image_phl: source_image,
                                    target_image_phl: target_image,
                                    source_landmarks_phl: source_landmarks,
                                    target_landmarks_phl: target_landmarks})

                example = _image_to_tfexample(source_image_data=source_image_string,
                                              target_image_data=target_image_string,
                                              source_landmarks_data=source_landmarks_string,
                                              target_landmarks_data=target_landmarks_string)

                training_tfrecord_writer.write(example.SerializeToString())

        with tf.python_io.TFRecordWriter(os.path.join(args.saved_dir, "validation.tfrecord")) as val_tfrecord_writer:
            for directory in tqdm(validation_directory_list):
                images = os.listdir(directory)
                if len(images) != 4:
                    continue

                source_image = cv2.imread(os.path.join(directory, "source_image.jpg"))[:, :, ::-1]
                target_image = cv2.imread(os.path.join(directory, "target_image.jpg"))[:, :, ::-1]
                source_landmarks = cv2.imread(os.path.join(directory, "source_landmarks.jpg"))[:, :, ::-1]
                target_landmarks = cv2.imread(os.path.join(directory, "target_landmarks.jpg"))[:, :, ::-1]

                source_image_string, target_image_string, source_landmarks_string, target_landmarks_string = \
                    sess.run([source_image_encoded, target_image_encoded, source_landmarks_encoded,
                              target_landmarks_encoded],
                             feed_dict={source_image_phl: source_image,
                                        target_image_phl: target_image,
                                        source_landmarks_phl: source_landmarks,
                                        target_landmarks_phl: target_landmarks})

                example = _image_to_tfexample(source_image_data=source_image_string,
                                              target_image_data=target_image_string,
                                              source_landmarks_data=source_landmarks_string,
                                              target_landmarks_data=target_landmarks_string)

                val_tfrecord_writer.write(example.SerializeToString())
