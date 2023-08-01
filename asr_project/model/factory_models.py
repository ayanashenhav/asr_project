from .base_model import BaseModel


def ModelsFactory(config) -> BaseModel:
    """ Return a Model object base on the model.name parameter in the config. """
    requested_model = config['name']

    if requested_model == 'CNN':
        from .cnn_model import CNNModel
        return CNNModel(config)
    elif requested_model == 'FastSpeech2':
        from .cnn_model import CNNModel
        return CNNModel(config)  # FastSpeech2(config)
    elif requested_model == 'Conformer':
        from .conformer_model import ConformerWrapper
        return ConformerWrapper(config)
    else:
        raise ValueError(f" [!] Unknown model name: {requested_model}")
