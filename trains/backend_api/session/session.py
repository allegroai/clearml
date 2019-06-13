import json as json_lib
import sys
import types
from socket import gethostname

import requests
import six
from pyhocon import ConfigTree
from requests.auth import HTTPBasicAuth

from .callresult import CallResult
from .defs import ENV_VERBOSE, ENV_HOST, ENV_ACCESS_KEY, ENV_SECRET_KEY
from .request import Request, BatchRequest
from .token_manager import TokenManager
from ..config import load
from ..utils import get_http_session_with_retry
from ..version import __version__


class LoginError(Exception):
    pass


class Session(TokenManager):
    """ TRAINS API Session class. """

    _AUTHORIZATION_HEADER = "Authorization"
    _WORKER_HEADER = "X-Trains-Worker"
    _ASYNC_HEADER = "X-Trains-Async"
    _CLIENT_HEADER = "X-Trains-Client"

    _async_status_code = 202
    _session_requests = 0
    _session_initial_timeout = (1.0, 10)
    _session_timeout = (5.0, None)

    # TODO: add requests.codes.gateway_timeout once we support async commits
    _retry_codes = [
        requests.codes.bad_gateway,
        requests.codes.service_unavailable,
        requests.codes.bandwidth_limit_exceeded,
        requests.codes.too_many_requests,
    ]

    @property
    def access_key(self):
        return self.__access_key

    @property
    def secret_key(self):
        return self.__secret_key

    @property
    def host(self):
        return self.__host

    @property
    def worker(self):
        return self.__worker

    def __init__(
        self,
        worker=None,
        api_key=None,
        secret_key=None,
        host=None,
        logger=None,
        verbose=None,
        initialize_logging=True,
        client=None,
        config=None,
        **kwargs
    ):

        if config is not None:
            self.config = config
        else:
            self.config = load()
            if initialize_logging:
                self.config.initialize_logging()

        token_expiration_threshold_sec = self.config.get(
            "auth.token_expiration_threshold_sec", 60
        )

        super(Session, self).__init__(
            token_expiration_threshold_sec=token_expiration_threshold_sec, **kwargs
        )

        self._verbose = verbose if verbose is not None else ENV_VERBOSE.get()
        self._logger = logger

        self.__access_key = api_key or ENV_ACCESS_KEY.get(
            default=self.config.get("api.credentials.access_key", None)
        )
        if not self.access_key:
            raise ValueError(
                "Missing access_key. Please set in configuration file or pass in session init."
            )

        self.__secret_key = secret_key or ENV_SECRET_KEY.get(
            default=self.config.get("api.credentials.secret_key", None)
        )
        if not self.secret_key:
            raise ValueError(
                "Missing secret_key. Please set in configuration file or pass in session init."
            )

        host = host or ENV_HOST.get(default=self.config.get("api.host"))
        if not host:
            raise ValueError("host is required in init or config")

        self.__host = host.strip("/")
        http_retries_config = self.config.get(
            "api.http.retries", ConfigTree()
        ).as_plain_ordered_dict()
        http_retries_config["status_forcelist"] = self._retry_codes
        self.__http_session = get_http_session_with_retry(**http_retries_config)

        self.__worker = worker or gethostname()

        self.__max_req_size = self.config.get("api.http.max_req_size")
        if not self.__max_req_size:
            raise ValueError("missing max request size")

        self.client = client or "api-{}".format(__version__)

        self.refresh_token()

    def _send_request(
        self,
        service,
        action,
        version=None,
        method="get",
        headers=None,
        auth=None,
        data=None,
        json=None,
        refresh_token_if_unauthorized=True,
    ):
        """ Internal implementation for making a raw API request.
            - Constructs the api endpoint name
            - Injects the worker id into the headers
            - Allows custom authorization using a requests auth object
            - Intercepts `Unauthorized` responses and automatically attempts to refresh the session token once in this
              case (only once). This is done since permissions are embedded in the token, and addresses a case where
              server-side permissions have changed but are not reflected in the current token. Refreshing the token will
              generate a token with the updated permissions.
        """
        host = self.host
        headers = headers.copy() if headers else {}
        headers[self._WORKER_HEADER] = self.worker
        headers[self._CLIENT_HEADER] = self.client

        token_refreshed_on_error = False
        url = (
            "{host}/v{version}/{service}.{action}"
            if version
            else "{host}/{service}.{action}"
        ).format(**locals())
        while True:
            res = self.__http_session.request(
                method, url, headers=headers, auth=auth, data=data, json=json,
                timeout=self._session_initial_timeout if self._session_requests < 1 else self._session_timeout,
            )
            if (
                refresh_token_if_unauthorized
                and res.status_code == requests.codes.unauthorized
                and not token_refreshed_on_error
            ):
                # it seems we're unauthorized, so we'll try to refresh our token once in case permissions changed since
                # the last time we got the token, and try again
                self.refresh_token()
                token_refreshed_on_error = True
                # try again
                continue
            if (
                res.status_code == requests.codes.service_unavailable
                and self.config.get("api.http.wait_on_maintenance_forever", True)
            ):
                self._logger.warn(
                    "Service unavailable: {} is undergoing maintenance, retrying...".format(
                        host
                    )
                )
                continue
            break
        self._session_requests += 1
        return res

    def send_request(
        self,
        service,
        action,
        version=None,
        method="get",
        headers=None,
        data=None,
        json=None,
        async_enable=False,
    ):
        """
        Send a raw API request.
        :param service: service name
        :param action: action name
        :param version: version number (default is the preconfigured api version)
        :param method: method type (default is 'get')
        :param headers: request headers (authorization and content type headers will be automatically added)
        :param json: json to send in the request body (jsonable object or builtin types construct. if used,
                     content type will be application/json)
        :param data: Dictionary, bytes, or file-like object to send in the request body
        :param async_enable: whether request is asynchronous
        :return: requests Response instance
        """
        headers = headers.copy() if headers else {}
        headers[self._AUTHORIZATION_HEADER] = "Bearer {}".format(self.token)
        if async_enable:
            headers[self._ASYNC_HEADER] = "1"
        return self._send_request(
            service=service,
            action=action,
            version=version,
            method=method,
            headers=headers,
            data=data,
            json=json,
        )

    def send_request_batch(
        self,
        service,
        action,
        version=None,
        headers=None,
        data=None,
        json=None,
        method="get",
    ):
        """
        Send a raw batch API request. Batch requests always use application/json-lines content type.
        :param service: service name
        :param action: action name
        :param version: version number (default is the preconfigured api version)
        :param headers: request headers (authorization and content type headers will be automatically added)
        :param json: iterable of json items (batched items, jsonable objects or builtin types constructs). These will
                     be sent as a multi-line payload in the request body.
        :param data: iterable of bytes objects (batched items). These will be sent as a multi-line payload in the
                     request body.
        :param method: HTTP method
        :return: requests Response instance
        """
        if not all(
            isinstance(x, (list, tuple, type(None), types.GeneratorType))
            for x in (data, json)
        ):
            raise ValueError("Expecting list, tuple or generator in 'data' or 'json'")

        if not data and not json:
            raise ValueError(
                "Missing data (data or json), batch requests are meaningless without it."
            )

        headers = headers.copy() if headers else {}
        headers["Content-Type"] = "application/json-lines"

        if data:
            req_data = "\n".join(data)
        else:
            req_data = "\n".join(json_lib.dumps(x) for x in json)

        cur = 0
        results = []
        while True:
            size = self.__max_req_size
            slice = req_data[cur : cur + size]
            if not slice:
                break
            if len(slice) < size:
                # this is the remainder, no need to search for newline
                pass
            elif slice[-1] != "\n":
                # search for the last newline in order to send a coherent request
                size = slice.rfind("\n") + 1
                # readjust the slice
                slice = req_data[cur : cur + size]
            res = self.send_request(
                method=method,
                service=service,
                action=action,
                data=slice,
                headers=headers,
                version=version,
            )
            results.append(res)
            if res.status_code != requests.codes.ok:
                break
            cur += size
        return results

    def validate_request(self, req_obj):
        """ Validate an API request against the current version and the request's schema """

        try:
            # make sure we're using a compatible version for this request
            # validate the request (checks required fields and specific field version restrictions)
            validate = req_obj.validate
        except AttributeError:
            raise TypeError(
                '"req_obj" parameter must be an backend_api.session.Request object'
            )

        validate()

    def send_async(self, req_obj):
        """
        Asynchronously sends an API request using a request object.
        :param req_obj: The request object
        :type req_obj: Request
        :return: CallResult object containing the raw response, response metadata and parsed response object.
        """
        return self.send(req_obj=req_obj, async_enable=True)

    def send(self, req_obj, async_enable=False, headers=None):
        """
        Sends an API request using a request object.
        :param req_obj: The request object
        :type req_obj: Request
        :param async_enable: Request this method be executed in an asynchronous manner
        :param headers: Additional headers to send with request
        :return: CallResult object containing the raw response, response metadata and parsed response object.
        """
        self.validate_request(req_obj)

        if isinstance(req_obj, BatchRequest):
            # TODO: support async for batch requests as well
            if async_enable:
                raise NotImplementedError(
                    "Async behavior is currently not implemented for batch requests"
                )

            json_data = req_obj.get_json()
            res = self.send_request_batch(
                service=req_obj._service,
                action=req_obj._action,
                version=req_obj._version,
                json=json_data,
                method=req_obj._method,
                headers=headers,
            )
            # TODO: handle multiple results in this case
            try:
                res = next(r for r in res if r.status_code != 200)
            except StopIteration:
                # all are 200
                res = res[0]
        else:
            res = self.send_request(
                service=req_obj._service,
                action=req_obj._action,
                version=req_obj._version,
                json=req_obj.to_dict(),
                method=req_obj._method,
                async_enable=async_enable,
                headers=headers,
            )

        call_result = CallResult.from_result(
            res=res,
            request_cls=req_obj.__class__,
            logger=self._logger,
            service=req_obj._service,
            action=req_obj._action,
            session=self,
        )

        return call_result

    def _do_refresh_token(self, old_token, exp=None):
        """ TokenManager abstract method implementation.
            Here we ignore the old token and simply obtain a new token.
        """
        verbose = self._verbose and self._logger
        if verbose:
            self._logger.info(
                "Refreshing token from {} (access_key={}, exp={})".format(
                    self.host, self.access_key, exp
                )
            )

        auth = HTTPBasicAuth(self.access_key, self.secret_key)
        try:
            data = {"expiration_sec": exp} if exp else {}
            res = self._send_request(
                service="auth",
                action="login",
                auth=auth,
                json=data,
                refresh_token_if_unauthorized=False,
            )
            try:
                resp = res.json()
            except ValueError:
                resp = {}
            if res.status_code != 200:
                msg = resp.get("meta", {}).get("result_msg", res.reason)
                raise LoginError(
                    "Failed getting token (error {} from {}): {}".format(
                        res.status_code, self.host, msg
                    )
                )
            if verbose:
                self._logger.info("Received new token")
            return resp["data"]["token"]
        except LoginError:
            six.reraise(*sys.exc_info())
        except Exception as ex:
            raise LoginError(str(ex))

    def __str__(self):
        return "{self.__class__.__name__}[{self.host}, {self.access_key}/{secret_key}]".format(
            self=self, secret_key=self.secret_key[:5] + "*" * (len(self.secret_key) - 5)
        )
