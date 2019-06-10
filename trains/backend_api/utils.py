import ssl
import sys

import requests
from requests.adapters import HTTPAdapter
## from requests.packages.urllib3.util.retry import Retry
from urllib3.util import Retry
from urllib3 import PoolManager
import six

if six.PY3:
    from functools import lru_cache
elif six.PY2:
    # python 2 support
    from backports.functools_lru_cache import lru_cache


@lru_cache()
def get_config():
    from ..backend_config import Config
    config = Config(verbose=False)
    config.reload()
    return config


class TLSv1HTTPAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        self.poolmanager = PoolManager(num_pools=connections,
                                       maxsize=maxsize,
                                       block=block,
                                       ssl_version=ssl.PROTOCOL_TLSv1_2)


def get_http_session_with_retry(
        total=0,
        connect=None,
        read=None,
        redirect=None,
        status=None,
        status_forcelist=None,
        backoff_factor=0,
        backoff_max=None,
        pool_connections=None,
        pool_maxsize=None):
    if not all(isinstance(x, (int, type(None))) for x in (total, connect, read, redirect, status)):
        raise ValueError('Bad configuration. All retry count values must be null or int')

    if status_forcelist and not all(isinstance(x, int) for x in status_forcelist):
        raise ValueError('Bad configuration. Retry status_forcelist must be null or list of ints')

    pool_maxsize = (
        pool_maxsize
        if pool_maxsize is not None
        else get_config().get('api.http.pool_maxsize', 512)
    )

    pool_connections = (
        pool_connections
        if pool_connections is not None
        else get_config().get('api.http.pool_connections', 512)
    )

    session = requests.Session()

    if backoff_max is not None:
        Retry.BACKOFF_MAX = backoff_max

    retry = Retry(
        total=total, connect=connect, read=read, redirect=redirect, status=status,
        status_forcelist=status_forcelist, backoff_factor=backoff_factor)

    adapter = TLSv1HTTPAdapter(max_retries=retry, pool_connections=pool_connections, pool_maxsize=pool_maxsize)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def get_response_cls(request_cls):
    """ Extract a request's response class using the mapping found in the module defining the request's service """
    for req_cls in request_cls.mro():
        module = sys.modules[req_cls.__module__]
        if hasattr(module, 'action_mapping'):
            return module.action_mapping[(request_cls._action, request_cls._version)][1]
        elif hasattr(module, 'response_mapping'):
            return module.response_mapping[req_cls]
    raise TypeError('no response class!')
