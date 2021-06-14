from app.model.buffer import Buffer, get_buffer_used_size
from app.model.response import ValuedResponse, Response, ExceptionResponse
from app.model.agent import load_agent_settings, save_agent_settings, get_loaded_agent
import tensorflow as tf
import numpy as np

agent = get_loaded_agent()


def remember(body: list):
    try:
        Buffer().store(body)
        return Response("added {} line(s) to buffer".format(len(body))).__dict__
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def set_weights(body):
    global agent
    try:
        agent.set_actor_weights(body)
        return ValuedResponse("updated actor weights", agent.get_actor_weights()).__dict__
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def get_weights():
    global agent
    return {"actor": agent.get_actor_weights(False), "target_actor": agent.get_actor_weights(True)}


def set_settings(body):
    global agent
    try:
        need_reload = save_agent_settings(body)
        agent.set_settings(body)
        if need_reload:
            agent = get_loaded_agent()
        return ValuedResponse("Updated agent settings", load_agent_settings()).__dict__
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def get_settings():
    global agent
    settings = agent.get_settings()
    return settings


def save(critic_only=False):
    global agent
    try:
        agent.save_models(critic_only)
        return Response("Successfully saved models").__dict__
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def learn(train_actor=True):
    global agent
    try:
        buffer_used_size = get_buffer_used_size()
        if buffer_used_size > 0:
            agent.learn(train_actor=train_actor)
            return agent.get_actor_weights()
        return Response("can't learn without experience", status="bad query").__dict__, 400
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def test_actor(body):
    global agent
    try:
        data = np.array(body)
        state = tf.convert_to_tensor(data[:, agent.used_states], dtype=tf.float32)
        return agent.actor(state, False).numpy().tolist()
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def test_critic(body):
    global agent
    try:
        data = np.array(body)
        states = tf.convert_to_tensor(data[:, np.concatenate((agent.used_states, [False]))], dtype=tf.float32)
        actions = tf.convert_to_tensor(data[:, -1:], dtype=tf.float32)
        return agent.critic(states, actions, training=False).numpy().tolist()
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500
