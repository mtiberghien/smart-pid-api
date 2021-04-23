from app.model.buffer import Buffer
from flask import request


def remember(body):
    return Buffer().store(body)
