import tensorflow as tf
import typer
from tensorflow.keras import layers

from modules.datasets import load_tfrecords
from modules.utils import load_yaml

AUTOTUNE = tf.data.experimental.AUTOTUNE


def main(config_path: str = 'config.yaml'):

    typer.secho("----- start training -----", fg=typer.colors.GREEN)

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

    input_shape = (config['input_width'], config['input_height'], config['input_chanel'])

    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(config['num_of_classes'], activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    model.fit(train_ds,
              steps_per_epoch=config['train_image_count'] // config['batch_size'],
              validation_data=test_ds,
              validation_steps=config['test_image_count'] // config['batch_size'],
              epochs=config['epochs'])

    typer.secho("----- end training -----", fg=typer.colors.GREEN)


if __name__ == '__main__':
    typer.run(main)
