
import tensorflow as tf
import tensorflow_datasets as tfds
import typer
from tensorflow.keras import layers

from modules.datasets import load_datasets
from modules.utils import load_yaml


def main(config_path: str = 'config.yaml'):

    # load config
    config = load_yaml('config.yaml')

    typer.secho(f"loaded config : {config_path}", fg=typer.colors.GREEN)

    # load dataset
    typer.secho(f"loaded dataset : {config['dataset']}", fg=typer.colors.GREEN)

    train_data, test_data = load_datasets(config)

    input_shape = (config['input_width'], config['input_height'], config['input_chanel'])

    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(101, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, epochs=5)


if __name__ == '__main__':
    typer.run(main)
