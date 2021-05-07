import tensorflow.keras as keras
from app.model.settings import data_dir
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras import layers
import os
from os import path
import tensorflow as tf


class ActorNetwork(keras.Model):
    def call(self, inputs, training=None, mask=None):
        output = self.mu(inputs, training=training)
        output = self.relu(-self.relu(self.max_action - output) + self.max_action - self.min_action) + self.min_action
        return output

    def get_config(self):
        return {"model_name": self.model_name, "checkpoint_dir": self.checkpoint_dir, "min_action": self.min_action,
                "max_action": self.max_action, "checkpoint_file": self.checkpoint_file}

    def __init__(self, inputs_shape, min_action, max_action,
                 name='actor', checkpoint_dir=data_dir):
        super(ActorNetwork, self).__init__(layers.Input(shape=inputs_shape))
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.min_action = min_action
        self.max_action = max_action
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name + '.h5')
        self.mu = Dense(1, use_bias=False, kernel_initializer=keras.initializers.Ones())
        self.relu = ReLU()

    def save_model(self):
        self.save_weights(self.checkpoint_file)

    def load(self):
        if path.exists(self.checkpoint_file):
            self.load_weights(self.checkpoint_file)
