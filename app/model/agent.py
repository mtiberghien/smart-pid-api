import os
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from app.model.buffer import get_buffer_sample
from app.model.settings import data_dir
from os import path
import json
import pickle


agent_settings_path = path.join(data_dir, 'agent.json')


def get_loaded_agent(load_models=True):
    settings = load_agent_settings()
    agent = Agent(**settings)
    if load_models:
        agent.build_models()
    return agent


def load_agent_settings():
    if path.exists(agent_settings_path):
        with open(agent_settings_path, mode='r') as file:
            settings = json.load(file)
        return settings
    return Agent().get_settings()


def save_agent_settings(settings):
    previous_agent = get_loaded_agent(False)
    previous_settings = previous_agent.get_settings()
    with open(agent_settings_path, mode='w') as file:
        json.dump(previous_settings | settings, file)
    new_agent = get_loaded_agent(False)
    if have_agents_structural_differences(previous_agent, new_agent):
        new_agent.reset_models()


def delete_file_if_exists(checkpoint_file):
    if path.exists(checkpoint_file):
        os.remove(checkpoint_file)


def get_normalized_tensor(tensor):
    result = tensor - tensor.mean(axis=0)
    result /= result.std(axis=0)
    return tf.cast(tf.where(tf.math.is_nan(result), tf.zeros_like(result), result), dtype=tf.float32)


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


class Agent:
    def __init__(self, use_p=True, use_i=True, use_d=True, use_iu=True, n_actions=2, alpha=0.001, beta=0.001,
                 gamma=0.99, tau=0.05, fc1=64, fc2=64, batch_size=256, min_action=0, max_action=100):
        self.n_actions = n_actions
        self.used_states = np.array([use_p, use_i, use_d, use_iu])
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_actions = n_actions
        self.min_action = min_action
        self.max_action = max_action
        self.fc1 = fc1
        self.fc2 = fc2
        self.n_inputs = self.get_n_inputs()
        self.actor_checkpoint_file = path.join(data_dir, 'actor_pid.h5')
        self.target_actor_checkpoint_file = path.join(data_dir, 'target_actor_pid.h5')
        self.critic_checkpoint_file = path.join(data_dir, 'critic_pid.h5')
        self.target_critic_checkpoint_file = path.join(data_dir, 'target_critic_pid.h5')
        self.actor_optimizer_file = path.join(data_dir, 'actor_optimizer.pkl')
        self.critic_optimizer_file = path.join(data_dir, 'critic_optimizer.pkl')
        self.actor_optimizer = Adam(learning_rate=self.alpha)
        self.critic_optimizer = Adam(learning_rate=self.beta)
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()

    def get_n_inputs(self):
        n_inputs = 0
        for used_state in self.used_states:
            if used_state:
                n_inputs += 1
        return n_inputs

    def get_settings(self):
        used_states = [bool(s) for s in iter(self.used_states)]
        return {"use_p": used_states[0], "use_i": used_states[1], "use_d": used_states[2],
                "use_iu": used_states[3], "n_actions": self.n_actions, "alpha": self.alpha, "beta": self.beta,
                "gamma": self.gamma, "tau": self.tau, "fc1": self.fc1, "fc2": self.fc2, "batch_size": self.batch_size,
                "min_action": self.min_action, "max_action": self.max_action}

    def update_network_parameters(self, tau_actor=None, tau_critic=None):
        if tau_actor is None:
            tau_actor = self.tau
        if tau_critic is None:
            tau_critic = self.tau
        update_target(self.target_actor.variables, self.actor.variables, tau_actor)
        update_target(self.target_critic.variables, self.critic.variables, tau_critic)

    def set_actor_weights(self, weights):
        weights = np.array(weights)[self.used_states].tolist()
        tf_weights = tf.convert_to_tensor([[[w] for w in weights]], dtype=tf.float32)
        self.actor.set_weights(tf_weights)
        self.target_actor.set_weights(tf_weights)
        self.actor.save_weights(self.actor_checkpoint_file)
        self.target_actor.save_weights(self.target_actor_checkpoint_file)

    def get_actor_weights(self, is_target=False):
        weights = (self.target_actor if is_target else self.actor).get_weights()
        weights = tf.squeeze(weights).numpy().tolist()
        result = [0, 0, 0, 0]
        i = 0
        if np.isscalar(weights):
            weights = [weights]
        for index, use_state in enumerate(self.used_states):
            if use_state:
                result[index] = weights[i]
                i += 1
        return result

    def save_optimizer(self, is_actor=True):
        file_path = self.actor_optimizer_file if is_actor else self.critic_optimizer_file
        optimizer = self.actor_optimizer if is_actor else self.critic_optimizer
        weights_values = tf.keras.backend.batch_get_value(optimizer.weights)
        with open(file_path, 'wb') as f:
            pickle.dump(weights_values, f)

    def load_optimizer(self, is_actor=True):
        file_path = self.actor_optimizer_file if is_actor else self.critic_optimizer_file
        if path.exists(file_path):
            optimizer = self.actor_optimizer if is_actor else self.critic_optimizer
            with open(file_path, 'rb') as f:
                weight_values = pickle.load(f)
            optimizer.set_weights(weight_values)

    def save_models(self, train_actor=True):
        if train_actor:
            self.actor.save_weights(self.actor_checkpoint_file)
            self.save_optimizer(True)
        self.critic.save_weights(self.critic_checkpoint_file)
        self.target_actor.save_weights(self.target_actor_checkpoint_file)
        self.target_critic.save_weights(self.target_critic_checkpoint_file)
        self.save_optimizer(False)

    def build_actor(self):
        actor = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.n_inputs,)),
            tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.Ones())
        ])
        return actor

    def build_critic(self):
        critic = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.n_inputs + self.n_actions,)),
            tf.keras.layers.Dense(self.fc1, activation='relu'),
            tf.keras.layers.Dense(self.fc2, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        return critic

    def load_model(self, checkpoint_file, is_target, is_actor):
        exists = path.exists(checkpoint_file)
        if is_actor:
            model = self.target_actor if is_target else self.actor
        else:
            model = self.target_critic if is_target else self.critic
        if exists:
            model.load_weights(checkpoint_file)

    def reset_models(self):
        self.actor = self.build_actor()
        self.target_actor = self.build_actor()
        self.critic = self.build_critic()
        self.target_critic = self.build_critic()
        self.save_models()

    def build_models(self):
        need_update_network = not path.exists(self.target_actor_checkpoint_file)

        self.load_model(self.actor_checkpoint_file, False, True)
        self.load_model(self.target_actor_checkpoint_file, True, True)
        self.load_model(self.critic_checkpoint_file, False, False)
        self.load_model(self.target_critic_checkpoint_file, True, False)
        self.load_optimizer(True)
        self.load_optimizer(False)

        if need_update_network:
            self.update_network_parameters(tau_actor=1, tau_critic=1)

    def get_actor_saturated(self, states, actor: tf.keras.Model, training=True):
        relu = tf.keras.layers.ReLU()
        output = actor(states, training=training)
        output = relu(-relu(self.max_action - output) + self.max_action - self.min_action) + self.min_action
        return output

    @tf.function
    def train(self, states, new_states, actions, rewards, steps, prev_steps, train_actor):
        with tf.GradientTape() as tape:
            target_actions = self.get_actor_saturated(new_states, self.target_actor, training=True)
            critic_value_ = self.target_critic(tf.concat([new_states, target_actions, steps], axis=1), training=True)
            critic_value = self.critic(tf.concat([states, actions, prev_steps], axis=1), training=True)
            target = rewards + self.gamma * critic_value_
            critic_loss = tf.math.reduce_mean(tf.math.square(target - critic_value))
        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_network_gradient,
                                                  self.critic.trainable_variables))
        if train_actor:
            with tf.GradientTape() as tape:
                new_policy_actions = self.get_actor_saturated(states, self.actor, training=True)
                actor_loss = -self.critic(tf.concat([states, new_policy_actions, prev_steps], axis=1), training=True)
                actor_loss = tf.math.reduce_mean(actor_loss)

            actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
            tf.debugging.check_numerics(actor_network_gradient, message="actor_loss:{}".format(actor_loss),
                                        name="training actor")
            self.actor_optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

    def learn(self, train_actor):
        used_batch_size, state, action, reward, new_state, step = get_buffer_sample(batch_size=self.batch_size)
        states = get_normalized_tensor(state[:, self.used_states])
        actions = self.get_actor_saturated(states, self.actor, training=False)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        new_states = get_normalized_tensor(new_state[:, self.used_states])
        steps = get_normalized_tensor(step)
        prev_steps = get_normalized_tensor(step-1)

        self.train(states, new_states, actions, rewards, steps, prev_steps, train_actor)
        self.update_network_parameters(tau_actor=(None if train_actor else 0))
        return used_batch_size


def have_agents_structural_differences(agent_1: Agent, agent_2: Agent):
    return agent_1.fc1 != agent_2.fc1 or agent_1.fc2 != agent_2.fc2 or \
           agent_1.n_inputs != agent_2.n_inputs or agent_1.n_actions != agent_2.n_actions
