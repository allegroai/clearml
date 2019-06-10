""" Configuration module. Uses backend_config to load system configuration. """
import logging
from os.path import expandvars, expanduser

from ..backend_api import load_config
from ..backend_config.bucket_config import S3BucketConfigurations

from .defs import *
from .remote import running_remotely_task_id as _running_remotely_task_id

config_obj = load_config(Path(__file__).parent)
config_obj.initialize_logging()
config = config_obj.get("sdk")
""" Configuration object reflecting the merged SDK section of all available configuration files """


def get_cache_dir():
    cache_base_dir = Path(
        expandvars(
            expanduser(
                config.get("storage.cache.default_base_dir") or DEFAULT_CACHE_DIR
            )
        )
    )
    return cache_base_dir


def get_config_for_bucket(base_url, extra_configurations=None):
    config_list = S3BucketConfigurations.from_config(config.get("aws.s3"))

    for configuration in extra_configurations or []:
        config_list.add_config(configuration)

    return config_list.get_config_by_uri(base_url)


def get_remote_task_id():
    return None


def running_remotely():
    return False


def get_log_to_backend(default=None):
    return LOG_TO_BACKEND_ENV_VAR.get(default=default)


def get_node_id(default=0):
    return NODE_ID_ENV_VAR.get(default=default)


def get_log_redirect_level():
    """ Returns which log level (and up) should be redirected to stderr. None means no redirection. """
    value = LOG_STDERR_REDIRECT_LEVEL.get()
    try:
        if value:
            return logging._checkLevel(value)
    except (ValueError, TypeError):
        pass


def dev_worker_name():
    return DEV_WORKER_NAME.get()
