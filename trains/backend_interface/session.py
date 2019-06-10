from abc import ABCMeta, abstractmethod

import six


class SendError(Exception):
    """ A session send() error class """
    @property
    def result(self):
        return self._result

    def __init__(self, result, *args, **kwargs):
        super(SendError, self).__init__(*args, **kwargs)
        self._result = result


@six.add_metaclass(ABCMeta)
class SessionInterface(object):
    """ Session wrapper interface providing a session property and a send convenience method """

    @property
    @abstractmethod
    def session(self):
        pass

    @abstractmethod
    def send(self, req, ignore_errors=False, raise_on_errors=True, async_enable=False):
        pass
