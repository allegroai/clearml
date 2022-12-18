from __future__ import print_function
import json as json_lib
import logging
import os
import sys
import types
from socket import gethostname
from time import sleep

import jwt
import requests
import six
from requests.auth import HTTPBasicAuth
from six.moves.urllib.parse import urlparse, urlunparse
from typing import List, Optional

from .callresult import CallResult
from .defs import (
    ENV_VERBOSE,
    ENV_HOST,
    ENV_ACCESS_KEY,
    ENV_SECRET_KEY,
    ENV_WEB_HOST,
    ENV_FILES_HOST,
    ENV_OFFLINE_MODE,
    ENV_AUTH_TOKEN,
    ENV_DISABLE_VAULT_SUPPORT,
    ENV_ENABLE_ENV_CONFIG_SECTION,
    ENV_ENABLE_FILES_CONFIG_SECTION,
    ENV_API_EXTRA_RETRY_CODES,
    ENV_API_DEFAULT_REQ_METHOD,
    MissingConfigError
)
from .request import Request, BatchRequest  # noqa: F401
from .token_manager import TokenManager
from ..utils import get_http_session_with_retry, urllib_log_warning_setup
from ...backend_config.defs import get_config_file
from ...debugging import get_logger
from ...debugging.log import resolve_logging_level
from ...utilities.pyhocon import ConfigTree, ConfigFactory
from ...version import __version__
from ...backend_config.utils import apply_files, apply_environment

try:
    from OpenSSL.SSL import Error as SSLError
except ImportError:
    from requests.exceptions import SSLError


class LoginError(Exception):
    pass


class MaxRequestSizeError(Exception):
    pass


class Session(TokenManager):
    """ ClearML API Session class. """

    _AUTHORIZATION_HEADER = "Authorization"
    _WORKER_HEADER = ("X-ClearML-Worker", "X-Trains-Worker", )
    _ASYNC_HEADER = ("X-ClearML-Async", "X-Trains-Async", )
    _CLIENT_HEADER = ("X-ClearML-Client", "X-Trains-Client", )

    _async_status_code = 202
    _session_requests = 0
    _session_initial_timeout = (3.0, 10.)
    _session_timeout = (10.0, 300.)
    _write_session_data_size = 15000
    _write_session_timeout = (300.0, 300.)
    _sessions_created = 0
    _ssl_error_count_verbosity = 2
    _offline_mode = ENV_OFFLINE_MODE.get()
    _offline_default_version = '2.9'

    _client = [(__package__.partition(".")[0], __version__)]

    api_version = '2.9'  # this default version should match the lowest api version we have under service
    max_api_version = '2.9'
    feature_set = 'basic'
    default_demo_host = "https://demoapi.demo.clear.ml"
    default_host = "https://api.clear.ml"
    default_web = "https://app.clear.ml"
    default_files = "https://files.clear.ml"
    default_key = ""  # "EGRTCO8JMSIGI6S39GTP43NFWXDQOW"
    default_secret = ""  # "x!XTov_G-#vspE*Y(h$Anm&DIc5Ou-F)jsl$PdOyj5wG1&E!Z8"
    force_max_api_version = None

    legacy_file_servers = ["https://files.community.clear.ml"]

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
    def auth_token(self):
        return self.__auth_token

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
        config=None,
        http_retries_config=None,
        **kwargs
    ):
        if config is not None:
            self.config = config
        else:
            from clearml.config import ConfigWrapper
            self.config = ConfigWrapper._init()

        token_expiration_threshold_sec = self.config.get(
            "auth.token_expiration_threshold_sec", 60
        )

        self._verbose = verbose if verbose is not None else ENV_VERBOSE.get()
        self._logger = logger
        if self._verbose and not self._logger:
            level = resolve_logging_level(ENV_VERBOSE.get(converter=str))
            self._logger = get_logger(level=level, stream=sys.stderr if level is logging.DEBUG else None)

        self.__auth_token = None

        self._update_default_api_method()

        if ENV_AUTH_TOKEN.get():
            self.__access_key = self.__secret_key = None
            self.__auth_token = ENV_AUTH_TOKEN.get()
            # if we use a token we override make sure we are at least 3600 seconds (1 hour)
            # away from the token expiration date, ask for a new one.
            token_expiration_threshold_sec = max(token_expiration_threshold_sec, 3600)
        elif not self._offline_mode:
            self.__access_key = api_key or ENV_ACCESS_KEY.get(
                default=(self.config.get("api.credentials.access_key", None) or self.default_key)
            )
            self.__secret_key = secret_key or ENV_SECRET_KEY.get(
                default=(self.config.get("api.credentials.secret_key", None) or self.default_secret)
            )

        # init the token manager
        super(Session, self).__init__(
            token_expiration_threshold_sec=token_expiration_threshold_sec, **kwargs
        )

        host = host or self.get_api_server_host(config=self.config)
        if not host:
            raise ValueError("ClearML host was not set, check your configuration file or environment variable")

        if not self._offline_mode and (not self.secret_key and not self.access_key and not self.__auth_token):
            raise MissingConfigError()

        self._ssl_error_count_verbosity = self.config.get(
            "api.ssl_error_count_verbosity", self._ssl_error_count_verbosity)

        self.__host = host.strip("/")
        http_retries_config = http_retries_config or self.config.get(
            "api.http.retries", ConfigTree()).as_plain_ordered_dict()

        http_retries_config["status_forcelist"] = self._get_retry_codes()
        http_retries_config["config"] = self.config
        self.__http_session = get_http_session_with_retry(**http_retries_config)
        self.__http_session.write_timeout = self._write_session_timeout
        self.__http_session.request_size_threshold = self._write_session_data_size

        self.__worker = worker or self.get_worker_host_name()

        self.__max_req_size = self.config.get("api.http.max_req_size", None)
        if not self.__max_req_size:
            raise ValueError("missing max request size")

        self.client = ", ".join("{}-{}".format(*x) for x in self._client)

        if self._offline_mode:
            return

        self.refresh_token()

        local_logger = self._LocalLogger(self._logger)

        # update api version from server response
        try:
            token_dict = TokenManager.get_decoded_token(self.token)
            api_version = token_dict.get('api_version')
            if not api_version:
                api_version = '2.2' if token_dict.get('env', '') == 'prod' else Session.api_version
            if token_dict.get('server_version'):
                if not any(True for c in Session._client if c[0] == 'clearml-server'):
                    Session._client.append(('clearml-server', token_dict.get('server_version'), ))

            Session.max_api_version = Session.api_version = str(api_version)
            Session.feature_set = str(token_dict.get('feature_set', self.feature_set) or "basic")
        except (jwt.DecodeError, ValueError):
            local_logger().warning(
                "Failed parsing server API level, defaulting to {}".format(Session.api_version))

        # now setup the session reporting, so one consecutive retries will show warning
        # we do that here, so if we have problems authenticating, we see them immediately
        # notice: this is across the board warning omission
        urllib_log_warning_setup(total_retries=http_retries_config.get('total', 0), display_warning_after=3)

        if self.force_max_api_version and self.check_min_api_version(self.force_max_api_version):
            Session.max_api_version = Session.api_version = str(self.force_max_api_version)

        # update only after we have max_api
        self.__class__._sessions_created += 1

        if self._load_vaults():
            from clearml.config import ConfigWrapper, ConfigSDKWrapper
            ConfigWrapper.set_config_impl(self.config)
            ConfigSDKWrapper.clear_config_impl()

        self._apply_config_sections(local_logger)

        self._update_default_api_method()

    def _update_default_api_method(self):
        if not ENV_API_DEFAULT_REQ_METHOD.get(default=None) and self.config.get("api.http.default_method", None):
            def_method = str(self.config.get("api.http.default_method", None)).strip()
            if def_method.upper() not in ("GET", "POST", "PUT"):
                raise ValueError(
                    "api.http.default_method variable must be 'get', 'post' or 'put' (any case is allowed)."
                )
            Request.def_method = def_method
            Request._method = Request.def_method

    def _get_retry_codes(self):
        # type: () -> List[int]
        retry_codes = set(self._retry_codes)

        extra = self.config.get("api.http.extra_retry_codes", [])
        if ENV_API_EXTRA_RETRY_CODES.get():
            extra = [s.strip() for s in ENV_API_EXTRA_RETRY_CODES.get().split(",") if s.strip()]

        for code in extra or []:
            try:
                retry_codes.add(int(code))
            except (ValueError, TypeError):
                print("Warning: invalid extra HTTP retry code detected: {}".format(code))

        if retry_codes.difference(self._retry_codes):
            print("Using extra HTTP retry codes {}".format(sorted(retry_codes.difference(self._retry_codes))))

        return list(retry_codes)

    def _load_vaults(self):
        # () -> Optional[bool]
        if not self.check_min_api_version("2.15") or self.feature_set == "basic":
            return

        if ENV_DISABLE_VAULT_SUPPORT.get():
            # (self._logger or get_logger()).debug("Vault support is disabled")
            return

        def parse(vault):
            # noinspection PyBroadException
            try:
                d = vault.get('data', None)
                if d:
                    r = ConfigFactory.parse_string(d)
                    if isinstance(r, (ConfigTree, dict)):
                        return r
            except Exception as e:
                (self._logger or get_logger()).warning("Failed parsing vault {}: {}".format(
                    vault.get("description", "<unknown>"), e))

        # noinspection PyBroadException
        try:
            # Use params and not data/json otherwise payload might be dropped if we're using GET with a strict firewall
            res = self.send_request("users", "get_vaults", params="enabled=true&types=config&types=config")
            if res.ok:
                vaults = res.json().get("data", {}).get("vaults", [])
                data = list(filter(None, map(parse, vaults)))
                if data:
                    self.config.set_overrides(*data)
                    return True
            elif res.status_code != 404:
                raise Exception(res.json().get("meta", {}).get("result_msg", res.text))
        except Exception as ex:
            (self._logger or get_logger()).warning("Failed getting vaults: {}".format(ex))

    def _apply_config_sections(self, local_logger):
        # type: (_LocalLogger) -> None  # noqa: F821
        default = self.config.get("sdk.apply_environment", False)
        if ENV_ENABLE_ENV_CONFIG_SECTION.get(default=default):
            try:
                keys = apply_environment(self.config)
                if keys:
                    print("Environment variables set from configuration: {}".format(keys))
            except Exception as ex:
                local_logger().warning("Failed applying environment from configuration: {}".format(ex))

        default = self.config.get("sdk.apply_files", default=False)
        if ENV_ENABLE_FILES_CONFIG_SECTION.get(default=default):
            try:
                apply_files(self.config)
            except Exception as ex:
                local_logger().warning("Failed applying files from configuration: {}".format(ex))

    def _send_request(
        self,
        service,
        action,
        version=None,
        method=None,
        headers=None,
        auth=None,
        data=None,
        json=None,
        refresh_token_if_unauthorized=True,
        params=None,
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
        if self._offline_mode:
            return None

        if not method:
            method = Request.def_method

        res = None
        host = self.host
        headers = headers.copy() if headers else {}
        for h in self._WORKER_HEADER:
            headers[h] = self.worker
        for h in self._CLIENT_HEADER:
            headers[h] = self.client

        token_refreshed_on_error = False
        url = (
            "{host}/v{version}/{service}.{action}"
            if version
            else "{host}/{service}.{action}"
        ).format(**locals())
        retry_counter = 0
        while True:
            if data and len(data) > self._write_session_data_size:
                timeout = self._write_session_timeout
            elif self._session_requests < 1:
                timeout = self._session_initial_timeout
            else:
                timeout = self._session_timeout
            try:
                if self._verbose and self._logger:
                    size = len(data or "")
                    if json and self._logger.level == logging.DEBUG:
                        size += len(json_lib.dumps(json))
                    self._logger.debug("%s: %s [%d bytes, %d headers]", method.upper(), url, size, len(headers or {}))
                res = self.__http_session.request(
                    method, url, headers=headers, auth=auth, data=data, json=json, timeout=timeout, params=params)
                if self._verbose and self._logger:
                    self._logger.debug("--> took %s", res.elapsed)
            # except Exception as ex:
            except SSLError as ex:
                retry_counter += 1
                # we should retry
                if retry_counter >= self._ssl_error_count_verbosity:
                    (self._logger or get_logger()).warning("SSLError Retrying {}".format(ex))
                sleep(0.1)
                continue

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
                retry_counter += 1
                continue
            if (
                res.status_code == requests.codes.service_unavailable
                and self.config.get("api.http.wait_on_maintenance_forever", True)
            ):
                (self._logger or get_logger()).warning(
                    "Service unavailable: {} is undergoing maintenance, retrying...".format(
                        host
                    )
                )
                retry_counter += 1
                continue
            break
        self._session_requests += 1
        return res

    def add_auth_headers(self, headers):
        headers[self._AUTHORIZATION_HEADER] = "Bearer {}".format(self.token)
        return headers

    def send_request(
        self,
        service,
        action,
        version=None,
        method=None,
        headers=None,
        data=None,
        json=None,
        async_enable=False,
        params=None,
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
        if not method:
            method = Request.def_method
        headers = self.add_auth_headers(
            headers.copy() if headers else {}
        )
        if async_enable:
            for h in self._ASYNC_HEADER:
                headers[h] = "1"
        return self._send_request(
            service=service,
            action=action,
            version=version,
            method=method,
            headers=headers,
            data=data,
            json=json,
            params=params,
        )

    def send_request_batch(
        self,
        service,
        action,
        version=None,
        headers=None,
        data=None,
        json=None,
        method=None,
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
            # Missing data (data or json), batch requests are meaningless without it.
            return None

        if not method:
            method = Request.def_method

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
            slice = req_data[cur: cur + size]
            if not slice:
                break
            if len(slice) < size:
                # this is the remainder, no need to search for newline
                pass
            elif slice[-1] != "\n":
                # search for the last newline in order to send a coherent request
                size = slice.rfind("\n") + 1
                # readjust the slice
                slice = req_data[cur: cur + size]
                if not slice:
                    raise MaxRequestSizeError('Error: {}.{} request exceeds limit {} > {} bytes'.format(
                        service, action, len(req_data), self.__max_req_size))
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

        if self._offline_mode:
            return None

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
            if res is not None:
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

    @classmethod
    def get_api_server_host(cls, config=None):
        if not config:
            from ...config import config_obj
            config = config_obj
        return ENV_HOST.get(default=(config.get("api.api_server", None) or
                                     config.get("api.host", None) or cls.default_host)).rstrip('/')

    @classmethod
    def get_app_server_host(cls, config=None):
        if not config:
            from ...config import config_obj
            config = config_obj

        # get from config/environment
        web_host = ENV_WEB_HOST.get(default=config.get("api.web_server", "")).rstrip('/')
        if web_host:
            return web_host

        # return default
        host = cls.get_api_server_host(config)
        if host == cls.default_host and cls.default_web:
            return cls.default_web

        # compose ourselves
        if '://demoapi.' in host:
            return host.replace('://demoapi.', '://demoapp.', 1)
        if '://api.' in host:
            return host.replace('://api.', '://app.', 1)

        parsed = urlparse(host)
        if parsed.port == 8008:
            return host.replace(':8008', ':8080', 1)

        raise ValueError('Could not detect ClearML web application server')

    @classmethod
    def get_files_server_host(cls, config=None):
        if not config:
            from ...config import config_obj
            config = config_obj
        # get from config/environment
        files_host = ENV_FILES_HOST.get(default=(config.get("api.files_server", ""))).rstrip('/')
        if files_host:
            return files_host

        # return default
        host = cls.get_api_server_host(config)
        if host == cls.default_host and cls.default_files:
            return cls.default_files

        # compose ourselves
        app_host = cls.get_app_server_host(config)
        parsed = urlparse(app_host)
        if parsed.port:
            parsed = parsed._replace(netloc=parsed.netloc.replace(':%d' % parsed.port, ':8081', 1))
        elif parsed.netloc.startswith('demoapp.'):
            parsed = parsed._replace(netloc=parsed.netloc.replace('demoapp.', 'demofiles.', 1))
        elif parsed.netloc.startswith('app.'):
            parsed = parsed._replace(netloc=parsed.netloc.replace('app.', 'files.', 1))
        else:
            parsed = parsed._replace(netloc=parsed.netloc + ':8081')

        return urlunparse(parsed)

    @classmethod
    def check_min_api_version(cls, min_api_version):
        """
        Return True if Session.api_version is greater or equal >= to min_api_version
        """
        # If no session was created, create a default one, in order to get the backend api version.
        if cls._sessions_created <= 0:
            if cls._offline_mode:
                # allow to change the offline mode version by setting ENV_OFFLINE_MODE to the required API version
                if cls.api_version != cls._offline_default_version:
                    offline_api = ENV_OFFLINE_MODE.get(converter=lambda x: x)
                    if offline_api:
                        try:
                            # check cast to float, but leave original str if we pass it.
                            # minimum version is 2.3
                            if float(offline_api) >= 2.3:
                                cls._offline_default_version = str(offline_api)
                        except ValueError:
                            pass
                    cls.max_api_version = cls.api_version = cls._offline_default_version
            else:
                # if the requested version is lower then the minimum we support,
                # no need to actually check what the server has, we assume it must have at least our version.
                if cls._version_tuple(cls.api_version) >= cls._version_tuple(str(min_api_version)):
                    return True

                # noinspection PyBroadException
                try:
                    cls()
                except Exception:
                    pass

        return cls._version_tuple(cls.api_version) >= cls._version_tuple(str(min_api_version))

    @classmethod
    def check_min_api_server_version(cls, min_api_version):
        """
        Return True if Session.max_api_version is greater or equal >= to min_api_version
        Notice this is the api version server reported, not the current SDK max supported api version
        """
        if cls.check_min_api_version(min_api_version):
            return True

        return cls._version_tuple(cls.max_api_version) >= cls._version_tuple(str(min_api_version))

    @classmethod
    def get_worker_host_name(cls):
        from ...config import dev_worker_name
        return dev_worker_name() or gethostname()

    @classmethod
    def get_clients(cls):
        return cls._client

    @staticmethod
    def _version_tuple(v):
        v = tuple(map(int, (v.split("."))))
        return v + (0,) * max(0, 3 - len(v))

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

        headers = None
        # use token only once (the second time the token is already built into the http session)
        if self.__auth_token:
            headers = dict(Authorization="Bearer {}".format(self.__auth_token))
            self.__auth_token = None

        auth = HTTPBasicAuth(self.access_key, self.secret_key) if self.access_key and self.secret_key else None
        res = None
        try:
            res = self._send_request(
                method=Request.def_method,
                service="auth",
                action="login",
                auth=auth,
                headers=headers,
                refresh_token_if_unauthorized=False,
                params={"expiration_sec": exp} if exp else {},
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

            # make sure we keep the token updated on the OS environment, so that child processes will have access.
            if ENV_AUTH_TOKEN.get():
                ENV_AUTH_TOKEN.set(resp["data"]["token"])

            return resp["data"]["token"]
        except LoginError:
            six.reraise(*sys.exc_info())
        except KeyError as ex:
            # check if this is a misconfigured api server (getting 200 without the data section)
            if res and res.status_code == 200:
                raise ValueError('It seems *api_server* is misconfigured. '
                                 'Is this the ClearML API server {} ?'.format(self.host))
            else:
                raise LoginError("Response data mismatch: No 'token' in 'data' value from res, receive : {}, "
                                 "exception: {}".format(res, ex))
        except Exception as ex:
            raise LoginError('Unrecognized Authentication Error: {} {}'.format(type(ex), ex))

    @staticmethod
    def __get_browser_token(webserver):
        # try to get the token if we are running inside a browser session (i.e. CoLab, Kaggle etc.)
        if not os.environ.get("JPY_PARENT_PID"):
            return None

        try:
            from google.colab import output  # noqa
            from google.colab._message import MessageError  # noqa
            from IPython import display  # noqa

            # must have cookie to same-origin: None for this one to work
            display.display(
                display.Javascript(
                    """
                    window._ApiKey = new Promise((resolve, reject) => {
                        const timeout = setTimeout(() => reject("Failed authenticating existing browser session"), 5000)
                        fetch("%s/api/auth.login", {
                          method: 'GET',
                          credentials: 'include'
                        })
                          .then((response) => resolve(response.json()))
                          .then((json) => {
                            clearTimeout(timeout);
                          }).catch((err) => {
                            clearTimeout(timeout);
                            reject(err);
                        });
                    });
                    """ % webserver.rstrip("/")
                ))

            response = output.eval_js("_ApiKey")
            if not response:
                return None
            result_code = response.get("meta", {}).get("result_code")
            token = response.get("data", {}).get("token")
        except:  # noqa
            return None

        if result_code != 200:
            raise ValueError(
                "Automatic authenticating failed, please login to {} and try again".format(webserver))

        return token

    def __str__(self):
        return "{self.__class__.__name__}[{self.host}, {self.access_key}/{secret_key}]".format(
            self=self, secret_key=self.secret_key[:5] + "*" * (len(self.secret_key) - 5)
        )

    class _LocalLogger:
        def __init__(self, local_logger):
            self.logger = local_logger

        def __call__(self):
            if not self.logger:
                self.logger = get_logger()
            return self.logger


def browser_login(clearml_server=None):
    # type: (Optional[str]) -> ()
    """
    Alternative authentication / login method, (instead of configuring ~/clearml.conf or Environment variables)
    ** Only applicable when running inside a browser session,
    for example Google Colab, Kaggle notebook, Jupyter Notebooks etc. **

    Notice: If called inside a python script, or when running with an agent, this function is ignored

    :param clearml_server: Optional, set the clearml server address, default: https://app.clear.ml
    """

    # check if we are running inside a Jupyter notebook of a sort
    if not os.environ.get("JPY_PARENT_PID"):
        return

    # if we are running remotely or in offline mode, skip login
    from clearml.config import running_remotely
    # noinspection PyProtectedMember
    if running_remotely():
        return

    # if we have working local configuration, nothing to do
    try:
        Session()
        return
    except:  # noqa
        pass

    # conform clearml_server address
    if clearml_server:
        if not clearml_server.lower().startswith("http"):
            clearml_server = "http://{}".format(clearml_server)

        parsed = urlparse(clearml_server)
        if parsed.port:
            parsed = parsed._replace(netloc=parsed.netloc.replace(':%d' % parsed.port, ':8008', 1))

        if parsed.netloc.startswith('demoapp.'):
            parsed = parsed._replace(netloc=parsed.netloc.replace('demoapp.', 'demoapi.', 1))
        elif parsed.netloc.startswith('app.'):
            parsed = parsed._replace(netloc=parsed.netloc.replace('app.', 'api.', 1))
        elif parsed.netloc.startswith('api.'):
            pass
        else:
            parsed = parsed._replace(netloc='api.' + parsed.netloc)

        clearml_server = urlunparse(parsed)

        # set for later usage
        ENV_HOST.set(clearml_server)

    token = None
    counter = 0
    clearml_app_server = Session.get_app_server_host()
    while not token:
        # try to get authentication toke
        try:
            # noinspection PyProtectedMember
            token = Session._Session__get_browser_token(clearml_app_server)
        except ValueError:
            token = None
        except Exception:  # noqa
            token = None
        # if we could not get a token, instruct the user to login
        if not token:
            if not counter:
                print(
                    "ClearML automatic browser login failed, please login or create a new account\n"
                    "To get started with ClearML: setup your own `clearml-server`, "
                    "or create a free account at {}\n".format(clearml_app_server)
                )
                print("Please login to {} , then press [Enter] to connect ".format(clearml_app_server), end="")
                input()
            elif counter < 1:
                print("Oh no we failed to connect \N{worried face}, "
                      "try to logout and login again - Press [Enter] to retry ", end="")
                input()
            else:
                print(
                    "\n"
                    "We cannot connect automatically (adblocker / incognito?) \N{worried face} \n"
                    "Please go to {}/settings/workspace-configuration \n"
                    "Then press \x1B[1m\x1B[48;2;26;30;44m\x1B[37m + Create new credentials \x1b[0m \n"
                    "And copy/paste your \x1B[1m\x1B[4mAccess Key\x1b[0m here: ".format(
                        clearml_app_server.lstrip("/")), end="")

                creds = input()
                if creds:
                    print(" Setting access key ")
                    ENV_ACCESS_KEY.set(creds.strip())

                print("Now copy/paste your \x1B[1m\x1B[4mSecret Key\x1b[0m here: ", end="")
                creds = input()
                if creds:
                    print(" Setting secret key ")
                    ENV_SECRET_KEY.set(creds.strip())

                if ENV_ACCESS_KEY.get() and ENV_SECRET_KEY.get():
                    # store in conf file for persistence in runtime
                    # noinspection PyBroadException
                    try:
                        with open(get_config_file(), "wt") as f:
                            f.write("api.credentials.access_key={}\napi.credentials.secret_key={}\n".format(
                                ENV_ACCESS_KEY.get(), ENV_SECRET_KEY.get()
                            ))
                    except Exception:
                        pass
                    break

            counter += 1

    print("")
    if counter:
        # these emojis actually requires python 3.6+
        # print("\nHurrah! \N{face with party horn and party hat} \N{confetti ball} \N{party popper}")
        print("\nHurrah! \U0001f973 \U0001f38a \U0001f389")

    if token:
        # set Token
        ENV_AUTH_TOKEN.set(token)

    if token or (ENV_ACCESS_KEY.get() and ENV_SECRET_KEY.get()):
        # verify token
        Session()
        # success
        print("\N{robot face} ClearML connected successfully - let's build something! \N{rocket}")
