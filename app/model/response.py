import traceback


class Response:
    def __init__(self, message, status="ok"):
        self.status = status
        self.message = message


class ValuedResponse(Response):
    def __init__(self, message, value, status="ok"):
        super(ValuedResponse, self).__init__(message, status)
        self.value = value


class ExceptionResponse(Response):
    def __init__(self, exception: Exception):
        super(ExceptionResponse, self).__init__(message=str(exception),
                                                status="Exception of type {}".format(exception.__class__))
        self.trace = traceback.format_exc()
