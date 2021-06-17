import os
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from app.model.buffer import get_buffer_sample
from app.model.settings import get_settings_file_path
from os import path
import json
from app.model.actor import ActorNetwork
from app.model.critic import CriticNetwork


def get_loaded_agent(load_models=True):
    settings = load_agent_settings()
    agent = Agent(**settings)
    if load_models:
        agent.build_models()
    return agent


def load_agent_settings():
    if path.exists(get_settings_file_path()):
        with open(get_settings_file_path(), mode='r') as file:
            settings = json.load(file)
        return settings
    return Agent().get_settings()


def save_agent_settings(settings):
    need_reload = False
    previous_agent = get_loaded_agent(False)
    previous_settings = previous_agent.get_settings()
    with open(get_settings_file_path(), mode='w') as file:
        json.dump(previous_settings | settings, file)
    new_agent = get_loaded_agent(False)
    if have_agents_structural_differences(previous_agent, new_agent):
        new_agent.save_models()
        need_reload = True
    return need_reload


def delete_file_if_exists(checkpoint_file):
    if path.exists(checkpoint_file):
        os.remove(checkpoint_file)


def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


class Agent:
    def __init__(self, use_p=True, use_i=True, use_d=True, use_iu=True, alpha=0.001, beta=0.001,
                 gamma=0.99, tau=0.05, fc1=64, fc2=64, batch_size=256, min_action=0, max_action=100):
        self.n_actions = 1
        self.used_states = np.array([use_p, use_i, use_d, use_iu])
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma
        self.min_action = min_action
        self.max_action = max_action
        self.fc1 = fc1
        self.fc2 = fc2
        self.n_inputs = self.get_n_inputs()
        self.actor_optimizer = Adam(learning_rate=self.alpha)
        self.critic_optimizer = Adam(learning_rate=self.beta)
        self.actor = ActorNetwork(self.n_inputs, name='actor',
                                  min_action=self.min_action, max_action=self.max_action)
        self.critic = CriticNetwork(self.n_inputs + self.n_actions, fc1_dims=self.fc1, fc2_dims=self.fc2, name='critic')
        self.target_actor = ActorNetwork(self.n_inputs, name='target_actor', min_action=self.min_action,
                                         max_action=self.max_action)
        self.target_critic = CriticNetwork(self.n_inputs + self.n_actions, fc1_dims=self.fc1, fc2_dims=self.fc2,
                                           name='target_critic')
        self.actor.model.compile(self.actor_optimizer)
        self.critic.model.compile(self.critic_optimizer)

    def get_n_inputs(self):
        n_inputs = 0
        for used_state in self.used_states:
            if used_state:
                n_inputs += 1
        return n_inputs

    def get_settings(self):
        used_states = [bool(s) for s in iter(self.used_states)]
        return {"use_p": used_states[0], "use_i": used_states[1], "use_d": used_states[2],
                "use_iu": used_states[3], "alpha": self.alpha, "beta": self.beta,
                "gamma": self.gamma, "tau": self.tau, "fc1": self.fc1, "fc2": self.fc2, "batch_size": self.batch_size,
                "min_action": self.min_action, "max_action": self.max_action}

    def set_settings(self, settings):
        if "use_p" in settings:
            self.used_states[0] = settings["use_p"]
        if "use_i" in settings:
            self.used_states[1] = settings["use_i"]
        if "use_d" in settings:
            self.used_states[2] = settings["use_d"]
        if "use_iu" in settings:
            self.used_states[3] = settings["use_iu"]
        self.n_inputs = self.get_n_inputs()
        if "alpha" in settings:
            self.gamma = settings["alpha"]
        if "beta" in settings:
            self.tau = settings["beta"]
        if "gamma" in settings:
            self.gamma = settings["gamma"]
        if "tau" in settings:
            self.tau = settings["tau"]
        if "fc1" in settings:
            self.fc1 = settings["fc1"]
        if "fc2" in settings:
            self.fc2 = settings["fc2"]
        if "batch_size" in settings:
            self.batch_size = settings["batch_size"]
        if "min_action" in settings:
            self.min_action = settings["min_action"]
            self.actor.min_action = self.min_action
            self.target_actor.min_action = self.min_action
        if "max_action" in settings:
            self.max_action = settings["max_action"]
            self.actor.max_action = self.max_action
            self.target_actor.max_action = self.max_action

    def update_network_parameters(self, tau_actor=None, tau_critic=None):
        if tau_actor is None:
            tau_actor = self.tau
        if tau_critic is None:
            tau_critic = self.tau
        update_target(self.target_actor.model.variables, self.actor.model.variables, tau_actor)
        update_target(self.target_critic.model.variables, self.critic.model.variables, tau_critic)

    def set_actor_weights(self, weights):
        weights = np.array(weights)[self.used_states].tolist()
        tf_weights = tf.convert_to_tensor([[[w] for w in weights]], dtype=tf.float32)
        self.actor.model.set_weights(tf_weights)
        self.target_actor.model.set_weights(tf_weights)
        self.actor.save()
        self.target_actor.save()

    def get_actor_weights(self, is_target=False):
        weights = (self.target_actor if is_target else self.actor).model.get_weights()
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

    def save_models(self, critic_only=False):
        if not critic_only:
            self.actor.save()
            self.target_actor.save()
        self.critic.save()
        self.target_critic.save()

    def build_models(self):
        need_update_network = not path.exists(self.target_actor.checkpoint_file)
        self.actor.load()
        self.critic.load()
        self.target_actor.load()
        self.target_critic.load()

        if need_update_network:
            self.update_network_parameters(tau_actor=1, tau_critic=1)

    def train(self, states, new_states, actions, rewards, train_actor):
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states, training=True)
            critic_value_ = self.target_critic(new_states, target_actions, training=True)
            critic_value = self.critic(states, actions, training=True)
            target = rewards + self.gamma * critic_value_
            critic_loss = tf.math.reduce_mean(tf.math.square(target - critic_value))
        critic_network_gradient = tape.gradient(critic_loss, self.critic.model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_network_gradient,
                                                  self.critic.model.trainable_variables))
        if train_actor:
            with tf.GradientTape() as tape:
                new_policy_actions = self.actor(states, training=True)
                actor_loss = self.critic(states, new_policy_actions, training=True)
                actor_loss = -tf.math.reduce_mean(actor_loss)

            actor_network_gradient = tape.gradient(actor_loss, self.actor.model.trainable_variables)
            tf.debugging.check_numerics(actor_network_gradient, message="actor_loss:{}".format(actor_loss))
            self.actor_optimizer.apply_gradients(zip(actor_network_gradient, self.actor.model.trainable_variables))

    def learn(self, train_actor):
        used_batch_size, state, action, reward, new_state = get_buffer_sample(batch_size=self.batch_size)
        states = tf.convert_to_tensor(state[:, self.used_states], dtype=tf.float32)
        actions = self.actor(states, training=False)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_state[:, self.used_states], dtype=tf.float32)

        self.train(states, new_states, actions, rewards, train_actor)
        self.update_network_parameters(tau_actor=(None if train_actor else 0))
        return used_batch_size


def have_agents_structural_differences(agent_1: Agent, agent_2: Agent):
    return agent_1.fc1 != agent_2.fc1 or agent_1.fc2 != agent_2.fc2 or \
           agent_1.n_inputs != agent_2.n_inputs
