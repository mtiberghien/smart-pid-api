from app.model.actor import ActorNetwork
from app.model.critic import CriticNetwork
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from tensorflow import keras
from app.model.buffer import get_buffer_sample
from app.model.settings import data_dir
from os import path
import json

agent_settings_path = path.join(data_dir, 'agent.json')


def get_loaded_agent():
    agent = Agent()
    agent.load_models()
    return agent


def load_agent_settings():
    if path.exists(agent_settings_path):
        with open(agent_settings_path, mode='r') as file:
            settings = json.load(file)
        return settings
    return {"min_action": 0, "max_action": 100}


def save_agent_settings(settings):
    with open(agent_settings_path, mode='w') as file:
        json.dump(settings, file)


class Agent:
    def __init__(self, n_inputs=4, n_actions=1, alpha=0.001, beta=0.002,
                 gamma=0.99, tau=0.05, fc1=64, fc2=64, batch_size=256, noise=0.1):
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.batch_size = batch_size
        self.noise = noise
        self.gamma = gamma
        self.n_actions = n_actions
        self.min_action = 0
        self.max_action = 100
        self.set_settings()

        self.actor = ActorNetwork((n_inputs,), self.min_action, self.max_action, name='actor_pid')
        self.critic = CriticNetwork((n_inputs + n_actions,), name='critic_pid',
                                    fc1_dims=fc1, fc2_dims=fc2)
        self.target_actor = ActorNetwork(n_inputs, self.min_action, self.max_action, name='target_actor_pid')
        self.target_critic = CriticNetwork(n_inputs + n_actions, name='target_critic_pid', fc1_dims=fc1,
                                           fc2_dims=fc2)

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))
        self.update_network_parameters(tau_actor=1, tau_critic=1)
        
        self.actor(tf.convert_to_tensor([np.zeros(n_inputs)]))
        self.target_actor(tf.convert_to_tensor([np.zeros(n_inputs)]))
        self.critic(tf.convert_to_tensor([np.zeros(n_inputs+n_actions)]))
        self.target_critic(tf.convert_to_tensor([np.zeros(n_inputs+n_actions)]))

    def set_settings(self):
        settings = load_agent_settings()
        if 'min_action' in settings:
            self.min_action = settings['min_action']
        if 'max_action' in settings:
            self.max_action = settings['max_action']

    def update_network_parameters(self, tau_actor=None, tau_critic=None):
        if tau_actor is None:
            tau_actor = self.tau
        if tau_critic is None:
            tau_critic = self.tau
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau_actor + (1 - tau_actor) * targets[i])
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau_critic + (1 - tau_critic) * targets[i])
        self.target_critic.set_weights(weights)

    def set_actor_weights(self, weights):
        tf_weights = tf.convert_to_tensor([[[w] for w in weights]], dtype=tf.float32)
        self.actor.set_weights(tf_weights)
        self.target_actor.set_weights(tf_weights)
        self.actor.save_model()
        self.target_actor.save_model()

    def get_actor_weights(self):
        return tf.squeeze(self.actor.get_weights()).numpy()

    def save_models(self):
        self.actor.save_model()
        self.critic.save_model()
        self.target_actor.save_model()
        self.target_critic.save_model()

    def load_models(self):
        self.actor.load()
        self.critic.load()
        self.target_actor.load()
        self.target_critic.load()

    def learn(self, update_actor):
        used_batch_size, state, action, reward, new_states, done = get_buffer_sample(batch_size=self.batch_size)
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states, training=True)
            critic_value_ = tf.squeeze(self.target_critic(tf.concat([new_states, target_actions], axis=1),
                                                          training=True), 1)
            critic_value = tf.squeeze(self.critic(tf.concat([states, actions], axis=1), training=True), 1)
            target = rewards + self.gamma * critic_value_ * (1 - done)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient,
                                                  self.critic.trainable_variables))
        if update_actor:
            with tf.GradientTape() as tape:
                new_policy_actions = self.actor(states, training=True)
                actor_loss = -self.critic(tf.concat([states, new_policy_actions], axis=1), training=True)
                actor_loss = tf.math.reduce_mean(actor_loss)

            actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters(tau_actor=(1 if update_actor else None))
        return used_batch_size
