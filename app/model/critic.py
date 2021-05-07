import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
import os
from os import path
from app.model.settings import data_dir


class CriticNetwork(keras.Model):
    def call(self, inputs, training=None, mask=None):
        output = self.fc1(inputs, training=training)
        output = self.fc2(output, training=training)
        output = self.q(output, training=training)
        return output

    def get_config(self):
        return {"fc1_dims": self.fc1_dims, "fc2_dims": self.fc2_dims, "model_name": self.model_name, "checkpoint_dir":
                self.checkpoint_dir, "checkpoint_file": self.checkpoint_file}

    def __init__(self, inputs_shape, fc1_dims=64, fc2_dims=64,
                 name='critic', checkpoint_dir=data_dir):
        super(CriticNetwork, self).__init__(layers.Input(shape=inputs_shape))
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name + '.h5')
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.q = Dense(1)

    def save_model(self):
        self.save_weights(self.checkpoint_file)

    def load(self):
        if path.exists(self.checkpoint_file):
            self.load_weights(self.checkpoint_file)
