from app.model.buffer import Buffer, get_buffer_used_size
from app.model.response import ValuedResponse, Response, ExceptionResponse
from app.model.agent import Agent, load_agent_settings, save_agent_settings, get_loaded_agent
import tensorflow as tf


def remember(body: list):
    try:
        Buffer().store(body)
        return Response("added {} line(s) to buffer".format(len(body))).__dict__
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def set_weights(body):
    try:
        agent = Agent()
        agent.set_actor_weights(body)
        return ValuedResponse("updated actor weights", agent.get_actor_weights().tolist()).__dict__
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def get_weights():
    agent = get_loaded_agent()
    return agent.get_actor_weights().tolist()


def set_settings(body):
    try:
        save_agent_settings(body)
        return ValuedResponse("Updated agent settings", load_agent_settings()).__dict__
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def get_settings():
    return load_agent_settings()


def learn():
    try:
        buffer_used_size = get_buffer_used_size()
        if buffer_used_size > 0:
            agent = get_loaded_agent()
            agent.learn(True)
            agent.save_models()
            return agent.get_actor_weights().tolist()
        return Response("can't learn without experience", status="bad query").__dict__, 400
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def test_actor(body):
    try:
        agent = get_loaded_agent()
        state = tf.convert_to_tensor(body, dtype=tf.float32)
        return agent.actor(state, training=False).numpy().tolist()
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def test_critic(body):
    try:
        agent = get_loaded_agent()
        state_action = tf.convert_to_tensor(body, dtype=tf.float32)
        return agent.critic(state_action, training=False).numpy().tolist()
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500
