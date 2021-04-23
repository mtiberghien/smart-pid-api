from app.model.buffer import save_buffer_settings
from app.model.buffer import get_buffer_settings


def set_buffer(body):
    save_buffer_settings(**body)
    buffer_settings = get_buffer_settings()
    return buffer_settings.__dict__
