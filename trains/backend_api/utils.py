import logging
import ssl
import sys

import requests
from requests.adapters import HTTPAdapter
## from requests.packages.urllib3.util.retry import Retry
from urllib3.util import Retry
from urllib3 import PoolManager
import six

from .session.defs import ENV_HOST_VERIFY_CERT

if six.PY3:
    from functools import lru_cache
elif six.PY2:
    # python 2 support
    from backports.functools_lru_cache import lru_cache


__disable_certificate_verification_warning = 0


@lru_cache()
def get_config():
    from ..config import config_obj
    return config_obj


def urllib_log_warning_setup(total_retries=10, display_warning_after=5):
    class RetryFilter(logging.Filter):
        last_instance = None

        def __init__(self, total, warning_after=5):
            super(RetryFilter, self).__init__()
            self.total = total
            self.display_warning_after = warning_after
            self.last_instance = self

        def filter(self, record):
            if record.args and len(record.args) > 0 and isinstance(record.args[0], Retry):
                retry_left = self.total - record.args[0].total
                return retry_left >= self.display_warning_after

            return True

    urllib3_log = logging.getLogger('urllib3.connectionpool')
    if urllib3_log:
        urllib3_log.removeFilter(RetryFilter.last_instance)
        urllib3_log.addFilter(RetryFilter(total_retries, display_warning_after))


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
    global __disable_certificate_verification_warning
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
    # update verify host certiface
    session.verify = ENV_HOST_VERIFY_CERT.get(default=get_config().get('api.verify_certificate', True))
    if not session.verify and __disable_certificate_verification_warning < 2:
        # show warning
        __disable_certificate_verification_warning += 1
        logging.getLogger('TRAINS').warning(
            msg='InsecureRequestWarning: Certificate verification is disabled! Adding '
                'certificate verification is strongly advised. See: '
                'https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings')
        # make sure we only do not see the warning
        import urllib3
        # noinspection PyBroadException
        try:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
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
