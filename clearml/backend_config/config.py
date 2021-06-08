from __future__ import print_function

import functools
import json
import os
import sys
import warnings
from fnmatch import fnmatch
from os.path import expanduser
from typing import Any

import six
from pathlib2 import Path
from ..utilities.pyhocon import ConfigTree, ConfigFactory
from pyparsing import (
    ParseFatalException,
    ParseException,
    RecursiveGrammarException,
    ParseSyntaxException,
)
from six.moves.urllib.parse import urlparse

from .bucket_config import S3BucketConfig
from .defs import (
    Environment,
    DEFAULT_CONFIG_FOLDER,
    LOCAL_CONFIG_PATHS,
    ENV_CONFIG_PATHS,
    LOCAL_CONFIG_FILES,
    LOCAL_CONFIG_FILE_OVERRIDE_VAR,
    ENV_CONFIG_PATH_OVERRIDE_VAR,
)
from .defs import is_config_file
from .entry import Entry, NotSet
from .errors import ConfigurationError
from .log import initialize as initialize_log, logger
from .utils import get_options

try:
    from typing import Text
except ImportError:
    # windows conda-less hack
    Text = Any


log = logger(__file__)


class ConfigEntry(Entry):
    logger = None

    def __init__(self, config, *keys, **kwargs):
        # type: (Config, Text, Any) -> None
        super(ConfigEntry, self).__init__(*keys, **kwargs)
        self.config = config

    def _get(self, key):
        # type: (Text) -> Any
        return self.config.get(key, NotSet)

    def error(self, message):
        # type: (Text) -> None
        log.error(message.capitalize())


class Config(object):
    """
    Represents a server configuration.
    If watch=True, will watch configuration folders for changes and reload itself.
    NOTE: will not watch folders that were created after initialization.
    """

    # used in place of None in Config.get as default value because None is a valid value
    _MISSING = object()

    def __init__(
        self,
        config_folder=None,
        env=None,
        verbose=True,
        relative_to=None,
        app=None,
        is_server=False,
        **_
    ):
        self._app = app
        self._verbose = verbose
        self._folder_name = config_folder or DEFAULT_CONFIG_FOLDER
        self._roots = []
        self._config = ConfigTree()
        self._env = env or os.environ.get("CLEARML_ENV", os.environ.get("TRAINS_ENV", Environment.default))
        self.config_paths = set()
        self.is_server = is_server

        if self._verbose:
            print("Config env:%s" % str(self._env))

        if not self._env:
            raise ValueError(
                "Missing environment in either init of environment variable"
            )
        if self._env not in get_options(Environment):
            raise ValueError("Invalid environment %s" % env)
        if relative_to is not None:
            self.load_relative_to(relative_to)

    @property
    def root(self):
        return self.roots[0] if self.roots else None

    @property
    def roots(self):
        return self._roots

    @roots.setter
    def roots(self, value):
        self._roots = value

    @property
    def env(self):
        return self._env

    def logger(self, path=None):
        return logger(path)

    def load_relative_to(self, *module_paths):
        def normalize(p):
            return Path(os.path.abspath(str(p))).with_name(self._folder_name)

        self.roots = list(map(normalize, module_paths))
        self.reload()

    def _reload(self):
        env = self._env
        config = self._config.copy()

        if self.is_server:
            env_config_paths = ENV_CONFIG_PATHS
        else:
            env_config_paths = []

        env_config_path_override = ENV_CONFIG_PATH_OVERRIDE_VAR.get()
        if env_config_path_override:
            env_config_paths = [expanduser(env_config_path_override)]

        # merge configuration from root and other environment config paths
        if self.roots or env_config_paths:
            config = functools.reduce(
                lambda cfg, path: ConfigTree.merge_configs(
                    cfg,
                    self._read_recursive_for_env(path, env, verbose=self._verbose),
                    copy_trees=True,
                ),
                self.roots + env_config_paths,
                config,
            )

        # merge configuration from local configuration paths
        if LOCAL_CONFIG_PATHS:
            config = functools.reduce(
                lambda cfg, path: ConfigTree.merge_configs(
                    cfg, self._read_recursive(path, verbose=self._verbose), copy_trees=True
                ),
                LOCAL_CONFIG_PATHS,
                config,
            )

        local_config_files = LOCAL_CONFIG_FILES
        local_config_override = LOCAL_CONFIG_FILE_OVERRIDE_VAR.get()
        if local_config_override:
            local_config_files = [expanduser(local_config_override)]

        # merge configuration from local configuration files
        if local_config_files:
            config = functools.reduce(
                lambda cfg, file_path: ConfigTree.merge_configs(
                    cfg,
                    self._read_single_file(file_path, verbose=self._verbose),
                    copy_trees=True,
                ),
                local_config_files,
                config,
            )

        config["env"] = env
        return config

    def replace(self, config):
        self._config = config

    def reload(self):
        self.replace(self._reload())

    def initialize_logging(self):
        logging_config = self._config.get("logging", None)
        if not logging_config:
            return False

        # handle incomplete file handlers
        deleted = []
        handlers = logging_config.get("handlers", {})
        for name, handler in list(handlers.items()):
            cls = handler.get("class", None)
            is_file = cls and "FileHandler" in cls
            if cls is None or (is_file and "filename" not in handler):
                deleted.append(name)
                del handlers[name]
            elif is_file:
                file = Path(handler.get("filename"))
                if not file.is_file():
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.touch()

        # remove dependency in deleted handlers
        root_logger = logging_config.get("root", None)
        loggers = list(logging_config.get("loggers", {}).values()) + (
            [root_logger] if root_logger else []
        )
        for logger in loggers:
            handlers = logger.get("handlers", None)
            if not handlers:
                continue
            logger["handlers"] = [h for h in handlers if h not in deleted]

        extra = None
        if self._app:
            extra = {"app": self._app}
        initialize_log(logging_config, extra=extra)
        return True

    def __getitem__(self, key):
        return self._config[key]

    def get(self, key, default=_MISSING):
        value = self._config.get(key, default)
        if value is self._MISSING:
            raise KeyError(
                "Unable to find value for key '{}' and default value was not provided.".format(
                    key
                )
            )
        return value

    def to_dict(self):
        return self._config.as_plain_ordered_dict()

    def as_json(self):
        return json.dumps(self.to_dict(), indent=2)

    def _read_recursive_for_env(self, root_path_str, env, verbose=True):
        root_path = Path(root_path_str)
        if root_path.exists():
            default_config = self._read_recursive(
                root_path / Environment.default, verbose=verbose
            )
            if (root_path / env) != (root_path / Environment.default):
                env_config = self._read_recursive(
                    root_path / env, verbose=verbose
                )  # None is ok, will return empty config
                config = ConfigTree.merge_configs(default_config, env_config, True)
            else:
                config = default_config
        else:
            config = ConfigTree()

        return config

    def _read_recursive(self, conf_root, verbose=True):
        conf = ConfigTree()
        if not conf_root:
            return conf
        conf_root = Path(conf_root)

        if not conf_root.exists():
            if verbose:
                print("No config in %s" % str(conf_root))
            return conf

        if verbose:
            print("Loading config from %s" % str(conf_root))
        for root, dirs, files in os.walk(str(conf_root)):

            rel_dir = str(Path(root).relative_to(conf_root))
            if rel_dir == ".":
                rel_dir = ""
            prefix = rel_dir.replace("/", ".")

            for filename in files:
                if not is_config_file(filename):
                    continue

                if prefix != "":
                    key = prefix + "." + Path(filename).stem
                else:
                    key = Path(filename).stem

                file_path = str(Path(root) / filename)

                conf.put(key, self._read_single_file(file_path, verbose=verbose))

        return conf

    @staticmethod
    def _read_single_file(file_path, verbose=True):
        if not file_path or not Path(file_path).is_file():
            return ConfigTree()

        if verbose:
            print("Loading config from file %s" % file_path)

        try:
            return ConfigFactory.parse_file(file_path)
        except ParseSyntaxException as ex:
            msg = "Failed parsing {0} ({1.__class__.__name__}): " \
                  "(at char {1.loc}, line:{1.lineno}, col:{1.column})".format(file_path, ex)
            six.reraise(
                ConfigurationError,
                ConfigurationError(msg, file_path=file_path),
                sys.exc_info()[2],
            )
        except (ParseException, ParseFatalException, RecursiveGrammarException) as ex:
            msg = "Failed parsing {0} ({1.__class__.__name__}): {1}".format(
                file_path, ex
            )
            six.reraise(ConfigurationError, ConfigurationError(msg), sys.exc_info()[2])
        except Exception as ex:
            print("Failed loading %s: %s" % (file_path, ex))
            raise

    def get_config_for_bucket(self, base_url, extra_configurations=None):
        """
        Get the credentials for an AWS S3 bucket from the config
        :param base_url: URL of bucket
        :param extra_configurations:
        :return: bucket config
        """

        warnings.warn(
            "Use backend_config.bucket_config.BucketList.get_config_for_uri",
            DeprecationWarning,
        )
        configs = S3BucketConfig.from_list(self.get("sdk.aws.s3.credentials", []))
        if extra_configurations:
            configs.extend(extra_configurations)

        def find_match(host=None, bucket=None):
            if not host and not bucket:
                raise ValueError("host or bucket required")
            try:
                if host:
                    res = {
                        config
                        for config in configs
                        if (config.host and fnmatch(host, config.host))
                        and (
                            not bucket
                            or not config.bucket
                            or fnmatch(bucket.lower(), config.bucket.lower())
                        )
                    }
                else:
                    res = {
                        config
                        for config in configs
                        if config.bucket
                        and fnmatch(bucket.lower(), config.bucket.lower())
                    }
                return next(iter(res))
            except StopIteration:
                pass

        parsed = urlparse(base_url)
        parts = Path(parsed.path.strip("/")).parts
        if parsed.netloc:
            # We have a netloc (either an actual hostname or an AWS bucket name).
            # First, we'll try with the netloc as host, but if we don't find anything, we'll try without a host and
            # with the netloc as the bucket name
            match = None
            if parts:
                # try host/bucket only if path parts contain any element
                match = find_match(host=parsed.netloc, bucket=parts[0])
            if not match:
                # no path parts or no config found for host/bucket, try netloc as bucket
                match = find_match(bucket=parsed.netloc)
        else:
            # No netloc, so we'll simply search by bucket
            match = find_match(bucket=parts[0])

        if match:
            return match

        non_aws_s3_host_suffix = ":9000"
        if parsed.netloc.endswith(non_aws_s3_host_suffix):
            host = parsed.netloc
            bucket = parts[0] if parts else None
        else:
            host = None
            bucket = parsed.netloc

        return S3BucketConfig(
            key=self.get("sdk.aws.s3.key", None),
            secret=self.get("sdk.aws.s3.secret", None),
            region=self.get("sdk.aws.s3.region", None),
            use_credentials_chain=self.get("sdk.aws.s3.use_credentials_chain", None),
            multipart=True,
            bucket=bucket,
            host=host,
        )
