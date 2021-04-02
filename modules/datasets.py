import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.data.ops.dataset_ops import MapDataset


def load_datasets(config):
    datasets = tfds.api.load(config['dataset'],
                             as_supervised=True,
                             data_dir='datasets',
                             download=config['is_download'])
    train, test = datasets['train'], datasets['validation']

    batch_size = config['batch_size']

    train_ds = train.map(lambda image, label: scale(image, label, config)).shuffle(1000, seed=0).batch(batch_size)
    test_ds = test.map(lambda image, label: scale(image, label, config)).batch(batch_size)

    return train_ds, test_ds


def scale(image, label, config):
    image = tf.cast(image, tf.float32)
    image /= 255.0
    image = tf.image.resize(image, [config['input_width'], config['input_height']])
    return image, label


def load_tfrecords(path: str, config: dict) -> MapDataset:
    """load tfrecords"""

    raw_dataset = tf.data.TFRecordDataset(path)

    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_function(example):

        feature = tf.io.parse_single_example(example, feature_description)

        image = tf.image.decode_jpeg(feature['image_raw'], channels=3)
        if config['is_resize']:
            image = tf.image.resize(image, [config['input_height'], config['input_width']])
        label = feature['label']

        return image, label

    parsed_dataset = raw_dataset.map(_parse_function)

    return parsed_dataset
