import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import VGG16, vgg16


def build_vgg16(config, inputs):

    preprocess = vgg16.preprocess_input

    x = preprocess(inputs)

    base = tf.keras.applications.vgg16.VGG16(input_shape=inputs.shape[1:],
                                             include_top=False,
                                             weights='imagenet')

    base.trainable = config['is_base_model_freeze']

    x = base(x)

    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(config['num_of_classes'], activation='softmax')(x)

    model = Model(inputs, outputs, name='VGG16')

    return model
