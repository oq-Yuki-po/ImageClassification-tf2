
from .vgg16 import build_vgg16


def build_model(config, inputs):

    if config['base_model'] == 'VGG16':
        model = build_vgg16(config, inputs)
    else:
        raise TypeError('base model error')

    return model
