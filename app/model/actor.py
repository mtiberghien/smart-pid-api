import tensorflow.keras as keras
import tensorflow as tf
from app.model.settings import data_dir
from tensorflow.keras.layers import Dense,ReLU
from tensorflow.keras import layers
import os
from os import path


class ActorNetwork(keras.Model):
    def get_config(self):
        return {"model_name": self.model_name, "checkpoint_dir": self.checkpoint_dir, "min_action": self.min_action,
                "max_action": self.max_action, "checkpoint_file": self.checkpoint_file}

    def __init__(self, inputs_shape, min_action, max_action,
                 name='actor', checkpoint_dir=data_dir):
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.min_action = min_action
        self.max_action = max_action
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name + '.h5')
        self.model_input = layers.Input(shape=inputs_shape)
        self.mu = Dense(1, activation=None, use_bias=False)(self.model_input)
        self.sat_mu = ReLU(max_value=self.max_action, threshold=self.min_action)(self.mu)
        super(ActorNetwork, self).__init__(self.model_input, self.sat_mu)

    def save(self):
        self.save_weights(self.checkpoint_file)

    def load(self):
        if path.exists(self.checkpoint_file):
            self.load_weights(self.checkpoint_file)
