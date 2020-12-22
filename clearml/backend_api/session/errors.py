class SessionError(Exception):
    pass


class AsyncError(SessionError):
    def __init__(self, msg, *args, **kwargs):
        super(AsyncError, self).__init__(msg, *args)
        for k, v in kwargs.items():
            setattr(self, k, v)


class TimeoutExpiredError(SessionError):
    pass


class ResultNotReadyError(SessionError):
    pass
