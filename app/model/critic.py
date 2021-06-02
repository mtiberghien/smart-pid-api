import tensorflow as tf
from tensorflow.keras.layers import Dense
import os
from app.model.settings import data_dir


class CriticNetwork:
    def __init__(self, n_inputs=4, fc1_dims=512, fc2_dims=512,
                 name='critic'):
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.n_inputs = n_inputs
        self.checkpoint_file = os.path.join(data_dir, self.model_name + '.h5')
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.n_inputs,)),
            tf.keras.layers.Dense(self.fc1, activation='relu'),
            tf.keras.layers.Dense(self.fc2, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def __call__(self, batch_state, batch_action, batch_step, training=True):
        q = self.model(tf.concat([batch_state, batch_action, batch_step], axis=1), training=training)
        return q

    def save(self):
        if self.model.optimizer is not None:
            self.model.save(self.checkpoint_file)
        else:
            self.model.save_weights(self.checkpoint_file)

    def load(self):
        if os.path.exists(self.checkpoint_file):
            if self.model.optimizer is not None:
                self.model = tf.keras.models.load_model(self.checkpoint_file)
            else:
                self.model.load_weights(self.checkpoint_file)
