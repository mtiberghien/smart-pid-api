import tensorflow as tf
import os
from app.model.settings import data_dir


class ActorNetwork:
    def __init__(self, n_inputs=3, name='actor', min_action=0, max_action=1):
        self.offset = min_action
        self.scale = max_action - min_action
        self.n_inputs = n_inputs
        self.model_name = name
        self.checkpoint_file = os.path.join(data_dir,
                                            self.model_name + '.h5')
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.n_inputs,)),
            tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.Ones())
        ])

    def __call__(self, state, training=True):
        mu = self.model(state, training=training)
        sat_mu = self.offset + self.scale * tf.keras.activations.sigmoid(mu)
        return sat_mu

    def save(self):
        if self.model.optimizer is not None:
            print('saving model:{}'.format(self.checkpoint_file))
            self.model.save(self.checkpoint_file)
        else:
            print('saving weights:{}'.format(self.checkpoint_file))
            self.model.save_weights(self.checkpoint_file)

    def load(self):
        if os.path.exists(self.checkpoint_file):
            if self.model.optimizer is not None:
                print('loading model:{}'.format(self.checkpoint_file))
                self.model = tf.keras.models.load_model(self.checkpoint_file)
            else:
                print('loading weights:{}'.format(self.checkpoint_file))
                self.model.load_weights(self.checkpoint_file)
