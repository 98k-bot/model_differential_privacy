import tensorflow.keras as keras
from tensorflow.keras import backend as K


class NoisySGD(keras.optimizers.SGD):
    def __init__(self, noise_eta=1, noise_gamma=4, **kwargs):
        super(NoisySGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.noise_eta = K.variable(noise_eta, name='noise_eta')
            self.noise_gamma = K.variable(noise_gamma, name='noise_gamma')

    def get_gradients(self, loss, params):
            grads = super(NoisySGD, self).get_gradients(loss, params)

            # Add decayed gaussian noise
            t = K.cast(self.iterations, K.dtype(grads[0]))
            variance = self.noise_eta / ((1 + t) ** self.noise_gamma)

            grads = [
                grad + K.random_normal(
                    grad.shape,
                    mean=0.0,
                    stddev=K.sqrt(variance),
                    dtype=K.dtype(grads[0])
                )
                for grad in grads
            ]

            return grads

    def get_config(self):
        config = {'noise_eta': float(K.get_value(self.noise_eta)),
                  'noise_gamma': float(K.get_value(self.noise_gamma))}
        base_config = super(NoisySGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))