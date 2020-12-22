import sys
from abc import ABCMeta, abstractmethod
from time import time

import jwt
import six


@six.add_metaclass(ABCMeta)
class TokenManager(object):

    @property
    def token_expiration_threshold_sec(self):
        return self.__token_expiration_threshold_sec

    @token_expiration_threshold_sec.setter
    def token_expiration_threshold_sec(self, value):
        self.__token_expiration_threshold_sec = value

    @property
    def req_token_expiration_sec(self):
        """ Token expiration sec requested when refreshing token """
        return self.__req_token_expiration_sec

    @req_token_expiration_sec.setter
    def req_token_expiration_sec(self, value):
        assert isinstance(value, (type(None), int))
        self.__req_token_expiration_sec = value

    @property
    def token_expiration_sec(self):
        return self.__token_expiration_sec

    @property
    def token(self):
        return self._get_token()

    @property
    def raw_token(self):
        return self.__token

    def __init__(
            self,
            token=None,
            req_token_expiration_sec=None,
            token_history=None,
            token_expiration_threshold_sec=60,
            **kwargs
    ):
        super(TokenManager, self).__init__()
        assert isinstance(token_history, (type(None), dict))
        self.token_expiration_threshold_sec = token_expiration_threshold_sec
        self.req_token_expiration_sec = req_token_expiration_sec
        self._set_token(token)

    def _calc_token_valid_period_sec(self, token, exp=None, at_least_sec=None):
        if token:
            try:
                exp = exp or self._get_token_exp(token)
                if at_least_sec:
                    at_least_sec = max(at_least_sec, self.token_expiration_threshold_sec)
                else:
                    at_least_sec = self.token_expiration_threshold_sec
                return max(0, (exp - time() - at_least_sec))
            except Exception:
                pass
        return 0

    @classmethod
    def _get_token_exp(cls, token):
        """ Get token expiration time. If not present, assume forever """
        return jwt.decode(token, verify=False).get('exp', sys.maxsize)

    def _set_token(self, token):
        if token:
            self.__token = token
            self.__token_expiration_sec = self._get_token_exp(token)
        else:
            self.__token = None
            self.__token_expiration_sec = 0

    def get_token_valid_period_sec(self):
        return self._calc_token_valid_period_sec(self.__token, self.token_expiration_sec)

    def _get_token(self):
        if self.get_token_valid_period_sec() <= 0:
            self.refresh_token()
        return self.__token

    @abstractmethod
    def _do_refresh_token(self, old_token, exp=None):
        pass

    def refresh_token(self):
        self._set_token(self._do_refresh_token(self.__token, exp=self.req_token_expiration_sec))
