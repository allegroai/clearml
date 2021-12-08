import abc

import requests.exceptions
import six
import jsonschema

from ..backend_api import Session, CallResult
from ..backend_api.session.session import MaxRequestSizeError
from ..backend_api.session.response import ResponseMeta
from ..backend_api.session import BatchRequest
from ..backend_api.session.defs import ENV_ACCESS_KEY, ENV_SECRET_KEY, ENV_OFFLINE_MODE

from ..config import config_obj
from ..config.defs import LOG_LEVEL_ENV_VAR
from ..debugging import get_logger
from .session import SendError, SessionInterface


class ValidationError(ValueError):
    pass


class InterfaceBase(SessionInterface):
    """ Base class for a backend manager class """
    _default_session = None
    _num_retry_warning_display = 1
    _offline_mode = ENV_OFFLINE_MODE.get()

    @property
    def session(self):
        return self._session

    @property
    def log(self):
        return self._log

    def __init__(self, session=None, log=None, **kwargs):
        super(InterfaceBase, self).__init__()
        self._session = session or self._get_default_session()
        self._log = log or self._create_log()

    def _create_log(self):
        log = get_logger(str(self.__class__.__name__))
        try:
            log.setLevel(LOG_LEVEL_ENV_VAR.get(default=log.level))
        except TypeError as ex:
            raise ValueError('Invalid log level defined in environment variable `%s`: %s' % (LOG_LEVEL_ENV_VAR, ex))
        return log

    @classmethod
    def _send(cls, session, req, ignore_errors=False, raise_on_errors=True, log=None, async_enable=False):
        """ Convenience send() method providing a standardized error reporting """
        if cls._offline_mode:
            return None

        num_retries = 0
        while True:
            error_msg = ''
            try:
                res = session.send(req, async_enable=async_enable)
                if res.meta.result_code in (200, 202) or ignore_errors:
                    return res

                if isinstance(req, BatchRequest):
                    error_msg = 'Action failed %s' % res.meta
                else:
                    error_msg = 'Action failed %s (%s)' \
                                % (res.meta, ', '.join('%s=%s' % p for p in req.to_dict().items()))
                if log:
                    log.error(error_msg)

            except requests.exceptions.BaseHTTPError as e:
                res = None
                if log and num_retries >= cls._num_retry_warning_display:
                    log.warning('Retrying, previous request failed %s: %s' % (str(type(req)), str(e)))
            except MaxRequestSizeError as e:
                res = CallResult(meta=ResponseMeta.from_raw_data(status_code=400, text=str(e)))
                error_msg = 'Failed sending: %s' % str(e)
            except requests.exceptions.ConnectionError:
                # We couldn't send the request for more than the retries times configure in the api configuration file,
                # so we will end the loop and raise the exception to the upper level.
                # Notice: this is a connectivity error and not a backend error.
                if raise_on_errors:
                    raise
                res = None
            except jsonschema.ValidationError as e:
                if log:
                    log.error(
                        'Field %s contains illegal schema: %s', '.'.join(e.path), str(e.message)
                    )
                if raise_on_errors:
                    raise ValidationError("Field %s contains illegal schema: %s" % ('.'.join(e.path), e.message))
                # We do not want to retry
                return None
            except Exception as e:
                res = None
                if log and num_retries >= cls._num_retry_warning_display:
                    log.warning('Retrying, previous request failed %s: %s' % (str(type(req)), str(e)))

            if res and res.meta.result_code <= 500:
                # Proper backend error/bad status code - raise or return
                if raise_on_errors:
                    raise SendError(res, error_msg)
                return res

            num_retries += 1

    def send(self, req, ignore_errors=False, raise_on_errors=True, async_enable=False):
        return self._send(session=self.session, req=req, ignore_errors=ignore_errors, raise_on_errors=raise_on_errors,
                          log=self.log, async_enable=async_enable)

    @classmethod
    def _get_default_session(cls):
        if not InterfaceBase._default_session:
            InterfaceBase._default_session = Session(
                initialize_logging=False,
                config=config_obj,
                api_key=ENV_ACCESS_KEY.get(),
                secret_key=ENV_SECRET_KEY.get(),
            )
        return InterfaceBase._default_session

    @classmethod
    def _set_default_session(cls, session):
        """
        Set a new default session to the system

        Warning: Use only for debug and testing
        :param session: The new default session
        """

        InterfaceBase._default_session = session

    @property
    def default_session(self):
        if hasattr(self, '_session'):
            return self._session
        return self._get_default_session()


@six.add_metaclass(abc.ABCMeta)
class IdObjectBase(InterfaceBase):

    def __init__(self, id, session=None, log=None, **kwargs):
        super(IdObjectBase, self).__init__(session, log, **kwargs)
        self._data = None
        self._id = None
        self.id = self.normalize_id(id)

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        should_reload = value is not None and self._id is not None and value != self._id
        self._id = value
        if should_reload:
            self.reload()

    @property
    def data(self):
        if self._data is None:
            self.reload()
        return self._data

    @abc.abstractmethod
    def _reload(self):
        pass

    def reload(self):
        if not self.id and not self._offline_mode:
            raise ValueError('Failed reloading %s: missing id' % type(self).__name__)
        # noinspection PyBroadException
        try:
            self._data = self._reload()
        except Exception:
            self.log.error("Failed reloading task {}".format(self.id))

    @classmethod
    def normalize_id(cls, id):
        return id.strip() if id else None

    @classmethod
    def resolve_id(cls, obj):
        if isinstance(obj, cls):
            return obj.id
        return obj
