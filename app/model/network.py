import os
from app.model.settings import get_data_dir
import tensorflow as tf


class Network:
    def __init__(self, name):
        self.model_name = name
        self.checkpoint_file = os.path.join(get_data_dir(),
                                            self.model_name + '.h5')
        self.best_checkpoint_file = os.path.join(get_data_dir(),
                                                 'best_' + self.model_name + '.h5')
        self.model = tf.keras.Sequential([])

    def save(self, is_best=False):
        if self.model.optimizer is not None:
            self.model.save(self.checkpoint_file)
            if is_best:
                self.model.save(self.best_checkpoint_file)
        else:
            self.model.save_weights(self.checkpoint_file)
            if is_best:
                self.model.save_weights(self.best_checkpoint_file)

    def load(self):
        if os.path.exists(self.checkpoint_file):
            if self.model.optimizer is not None:
                self.model = tf.keras.models.load_model(self.checkpoint_file)
            else:
                self.model.load_weights(self.checkpoint_file)
