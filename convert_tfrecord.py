from __future__ import annotations

import pathlib

import tensorflow as tf
import tqdm
import typer

from modules.utils import load_yaml


def main(config_path: str = 'config.yaml'):

    typer.secho("----- starting convert to tfrecors -----", fg=typer.colors.GREEN)

    config = load_yaml('config.yaml')

    typer.secho(f"loaded config : {config_path}", fg=typer.colors.GREEN)

    typer.secho(f"target train dataset : {config['train_row_images']}", fg=typer.colors.GREEN)
    typer.secho(f"target test dataset : {config['test_row_images']}", fg=typer.colors.GREEN)

    p_train_images = pathlib.Path(config['train_row_images'])
    p_test_images = pathlib.Path(config['test_row_images'])

    labels = {}

    for index, i in enumerate(list(p_train_images.iterdir())):
        labels[i.stem] = index

    typer.secho(f"----- convert train dataset -----", fg=typer.colors.GREEN)
    train_images = make_image_labels(p_train_images, labels, config)
    train_record_file = f'{config["saved_dir"]}/{config["tfrecod_train_name"]}'
    write_tfrecords(train_record_file, train_images)

    typer.secho(f"----- convert test dataset -----", fg=typer.colors.GREEN)
    test_images = make_image_labels(p_test_images, labels, config)
    test_record_file = f'{config["saved_dir"]}/{config["tfrecod_test_name"]}'
    write_tfrecords(test_record_file, test_images)

    typer.secho(f"----- finishing convert to tfrecors -----", fg=typer.colors.GREEN)


def write_tfrecords(file_name: str, image_labels: dict):
    """write tfrecords"""
    with tf.io.TFRecordWriter(file_name) as writer:
        for image_path, label in tqdm.tqdm(image_labels.items()):
            image_string = open(image_path, 'rb').read()
            tf_example = image_example(image_string, label)
            writer.write(tf_example.SerializeToString())


def make_image_labels(p_path: pathlib.Path, labels: dict, config: dict):
    """make image labels"""
    image_labels = {}

    for i in list(p_path.iterdir()):
        label_name = i.stem
        label = labels[label_name]
        p_label_images = pathlib.Path(f"{str(i)}")
        for l in list(p_label_images.glob(f"**/*.{config['image_extension']}")):
            image_labels[str(l)] = label

    return image_labels


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string: str | bytes, label: int):
    """convert example"""
    feature = {
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


if __name__ == '__main__':
    typer.run(main)
