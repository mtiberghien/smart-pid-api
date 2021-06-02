from app.model.buffer import Buffer, get_buffer_used_size
from app.model.response import ValuedResponse, Response, ExceptionResponse
from app.model.agent import load_agent_settings, save_agent_settings, get_loaded_agent
import tensorflow as tf
import numpy as np


def remember(body: list):
    try:
        Buffer().store(body)
        return Response("added {} line(s) to buffer".format(len(body))).__dict__
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def set_weights(body):
    try:
        agent = get_loaded_agent()
        agent.set_actor_weights(body)
        return ValuedResponse("updated actor weights", agent.get_actor_weights()).__dict__
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def get_weights():
    agent = get_loaded_agent()
    return {"actor": agent.get_actor_weights(False), "target_actor": agent.get_actor_weights(True)}


def set_settings(body):
    try:
        save_agent_settings(body)
        return ValuedResponse("Updated agent settings", load_agent_settings()).__dict__
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def get_settings():
    settings = get_loaded_agent(False).get_settings()
    return settings


def learn(train_actor=True):
    try:
        buffer_used_size = get_buffer_used_size()
        if buffer_used_size > 0:
            agent = get_loaded_agent()
            agent.learn(train_actor=train_actor)
            agent.save_models(train_actor)
            return agent.get_actor_weights()
        return Response("can't learn without experience", status="bad query").__dict__, 400
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def test_actor(body):
    try:
        agent = get_loaded_agent()
        data = np.array(body)
        state = tf.convert_to_tensor(data[:, agent.used_states], dtype=tf.float32)
        return agent.actor(state, False).numpy().tolist()
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def test_critic(body):
    try:
        agent = get_loaded_agent()
        data = np.array(body)
        states = data[:, np.concatenate((agent.used_states, [False, False]))]
        actions = data[:, -2:-1]
        steps = data[:, -1:]
        return agent.critic(states, actions, steps, training=False).numpy().tolist()
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500
