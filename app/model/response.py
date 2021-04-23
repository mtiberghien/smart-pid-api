class Response:
    def __init__(self,message, status = "ok"):
        self.status = status
        self.message = message


class BufferStorageResponse(Response):
    def __init__(self, message, value, status="ok"):
        super().__init__(message, status)
        self.value = value
