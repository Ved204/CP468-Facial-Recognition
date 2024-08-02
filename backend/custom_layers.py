from tensorflow.keras.layers import BatchNormalization

class CustomBatchNormalization(BatchNormalization):
    def __init__(self, **kwargs):
        # Convert axis from list to int if necessary
        if 'axis' in kwargs and isinstance(kwargs['axis'], list):
            kwargs['axis'] = kwargs['axis'][0]
        super(CustomBatchNormalization, self).__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        # Ensure axis is an int
        config['axis'] = config['axis'][0] if isinstance(config['axis'], list) else config['axis']
        return cls(**config)
