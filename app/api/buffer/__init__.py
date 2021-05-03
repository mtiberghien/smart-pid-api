from app.model.buffer import reset_buffer, get_buffer_sample
from app.model.response import Response


def reset():
    reset_buffer()
    return Response("memory buffer data was deleted").__dict__


def get_sample(batch_size=32):
    used_batch_size, states, actions, rewards, new_states, are_done = get_buffer_sample(batch_size)
    return {"states": states.tolist(), "actions": actions.tolist(), "rewards": rewards.tolist(),
            "new_states": new_states.tolist(), "are_done": are_done.tolist()}
