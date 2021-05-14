from app.model.buffer import reset_buffer, get_buffer_sample, get_buffer_settings, save_buffer_settings
from app.model.response import Response, ExceptionResponse


def reset():
    try:
        reset_buffer()
        return Response("memory buffer data was deleted").__dict__
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def get_sample(batch_size=32):
    used_batch_size, states, actions, rewards, new_states = get_buffer_sample(batch_size)
    return {"states": states.tolist(), "actions": actions.tolist(), "rewards": rewards.tolist(),
            "new_states": new_states.tolist()}


def set_settings(body):
    try:
        save_buffer_settings(**body)
        buffer_settings = get_buffer_settings()
        return buffer_settings.__dict__
    except Exception as e:
        return ExceptionResponse(e).__dict__, 500


def get_settings():
    return get_buffer_settings().__dict__
