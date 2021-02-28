import tensorflow as tf
import tensorflow_datasets as tfds


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
