import tensorflow as tf
from app.model.network import Network


class CriticNetwork(Network):
    def __init__(self, n_inputs=4, fc1_dims=512, fc2_dims=512,
                 name='critic'):
        super().__init__(name)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_inputs = n_inputs
        self.model.add(tf.keras.layers.InputLayer(input_shape=(self.n_inputs,)))
        self.model.add(tf.keras.layers.Dense(self.fc1_dims, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.fc2_dims, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1))

    def __call__(self, batch_state, batch_action, training=True):
        q = self.model(tf.concat([batch_state, batch_action], axis=1), training=training)
        return q
