import cv2
import tensorflow as tf


keys_to_features = {"source_image": tf.FixedLenFeature([], tf.string),
                    "target_image": tf.FixedLenFeature([], tf.string),
                    "source_landmarks": tf.FixedLenFeature([], tf.string),
                    "target_landmarks": tf.FixedLenFeature([], tf.string)}


def _parse_fn(data_record):
    features = keys_to_features
    sample = tf.parse_single_example(data_record, features)

    source_image = tf.image.decode_jpeg(sample["source_image"])
    source_image = tf.cast(source_image, dtype=tf.float32)
    target_image = tf.image.decode_jpeg(sample["target_image"])
    target_image = tf.cast(target_image, dtype=tf.float32)
    source_landmarks = tf.image.decode_jpeg(sample["source_landmarks"])
    source_landmarks = tf.cast(source_landmarks, dtype=tf.float32)
    target_landmarks = tf.image.decode_jpeg(sample["target_landmarks"])
    target_landmarks = tf.cast(target_landmarks, dtype=tf.float32)

    return source_image, target_image, source_landmarks, target_landmarks


def get_batch(tfrecord_path, batch_size, num_epochs=100):
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    dataset = dataset.map(_parse_fn)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(2000)
    dataset = dataset.repeat(num_epochs)

    iterator = dataset.make_one_shot_iterator()
    source_image, target_image, source_landmarks, target_landmarks = iterator.get_next()

    return source_image, target_image, source_landmarks, target_landmarks



#dataset = tf.data.TFRecordDataset([r"K:\VoxCeleb2\tfrecord\training.tfrecord"])
#dataset = dataset.map(_parse_fn)
#dataset = dataset.batch(1)
#dataset = dataset.shuffle(100)
#dataset = dataset.repeat(1)
#
#iterator = dataset.make_one_shot_iterator()
#source_image, target_image, source_landmarks, target_landmarks = iterator.get_next()
#
#with tf.Session() as sess:
#    try:
#        while True:
#            source, target, source_lm, target_lm = sess.run([source_image, target_image, source_landmarks, target_landmarks])
#
#            source = source[0][:, :, ::-1]
#            target = target[0][:, :, ::-1]
#            source_lm = source_lm[0][:, :, ::-1]
#            target_lm = target_lm[0][:, :, ::-1]
#
#            cv2.imshow("Source image record", source)
#            cv2.imshow("Target image record", target)
#            cv2.imshow("Source landmarks record", source_lm)
#            cv2.imshow("Target landmarks record", target_lm)
#            cv2.waitKey(2000)
#
#    except:
#        print("Finished")
#        pass
