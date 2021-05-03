from app.model.buffer import Buffer, get_buffer_used_size
from app.model.response import ValuedResponse, Response
from app.model.agent import Agent, load_agent_settings, save_agent_settings


def remember(body):
    Buffer().store(body)
    return ValuedResponse("updated buffer memory", body).__dict__


def set_weights(body):
    agent = Agent()
    agent.set_actor_weights(body)
    return ValuedResponse("updated actor weights", agent.get_actor_weights().tolist()).__dict__


def get_weights():
    agent = Agent()
    agent.load_models()
    return agent.get_actor_weights().tolist()


def set_settings(body):
    save_agent_settings(body)
    return ValuedResponse("Updated agent settings", load_agent_settings()).__dict__


def get_settings():
    return load_agent_settings()


def learn():
    buffer_used_size = get_buffer_used_size()
    if buffer_used_size > 0:
        update_actor = buffer_used_size >= Buffer().settings.mem_size/2.
        agent = Agent()
        agent.load_models()
        used_batch_size = agent.learn(update_actor)
        agent.save_models()
        return ValuedResponse("learned using {} items batch random sample from buffer".format(used_batch_size),
                              agent.get_actor_weights().tolist()).__dict__
    return Response("can't learn without experience", status="bad query").__dict__, 400
