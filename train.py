import os

import tensorflow as tf
import typer
from tensorflow.keras import layers

from models import build_model
from modules.datasets import load_tfrecords
from modules.utils import load_yaml

AUTOTUNE = tf.data.experimental.AUTOTUNE


def main(config_path: str = 'config.yaml'):

    typer.secho("--------------- start training ---------------", fg=typer.colors.GREEN)

    # load config
    config = load_yaml('config.yaml')

    typer.secho(f"loaded config : {config_path}", fg=typer.colors.GREEN)

    # load dataset
    typer.secho(f"loaded train dataset : {config['train_data']}", fg=typer.colors.GREEN)
    typer.secho(f"loaded test dataset : {config['test_data']}", fg=typer.colors.GREEN)

    train_ds = load_tfrecords(config['train_data'], config)
    test_ds = load_tfrecords(config['test_data'], config)

    train_ds = (train_ds.shuffle(buffer_size=config['train_image_count'] // 10)
                .repeat()
                .batch(config['batch_size'])
                .prefetch(buffer_size=AUTOTUNE))

    test_ds = (test_ds.repeat()
               .batch(config['batch_size'])
               .prefetch(buffer_size=AUTOTUNE))

    typer.secho("--------------- build model ---------------", fg=typer.colors.GREEN)

    input_shape = (config['input_width'], config['input_height'], config['input_chanel'])

    typer.secho(f"base model : {config['base_model']}", fg=typer.colors.GREEN)

    inputs = tf.keras.Input(shape=input_shape)

    model = build_model(config, inputs)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    log_dir = f'logs/{config["base_model"]}'
    os.mkdir(log_dir)

    tfboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model.fit(train_ds,
              steps_per_epoch=config['train_image_count'] // config['batch_size'],
              validation_data=test_ds,
              validation_steps=config['test_image_count'] // config['batch_size'],
              epochs=config['epochs'],
              callbacks=[tfboard_callbacks])

    typer.secho("--------------- end training ---------------", fg=typer.colors.GREEN)


if __name__ == '__main__':
    typer.run(main)
