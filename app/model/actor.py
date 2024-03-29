import tensorflow as tf
from app.model.network import Network


class ActorNetwork(Network):
    def __init__(self, n_inputs=4, name='actor', min_action=0, max_action=1):
        super().__init__(name)
        self.min_action = min_action
        self.max_action = max_action
        self.n_inputs = n_inputs
        self.model_name = name
        self.model.add(tf.keras.layers.InputLayer(input_shape=(self.n_inputs,)))
        self.model.add(tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.Ones()))

    def __call__(self, state, training=True):
        mu = self.model(state, training=training)
        sat_mu = tf.keras.activations.relu(-tf.keras.activations.relu(self.max_action - mu) + self.max_action
                                           - self.min_action) + self.min_action
        return sat_mu
