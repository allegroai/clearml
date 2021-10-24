from os.path import expanduser, expandvars, exists
from pathlib2 import Path

from .environment import EnvEntry


ENV_VAR = 'CLEARML_ENV'
""" Name of system environment variable that can be used to specify the config environment name """


DEFAULT_CONFIG_FOLDER = 'config'
""" Default config folder to search for when loading relative to a given path """


ENV_CONFIG_PATHS = [
]


""" Environment-related config paths """


LOCAL_CONFIG_PATHS = [
    # '/etc/opt/trains',               # used by servers for docker-generated configuration
    # expanduser('~/.trains/config'),
]
""" Local config paths, not related to environment """


LOCAL_CONFIG_FILES = [
    expanduser('~/trains.conf'),    # used for workstation configuration (end-users, workers)
    expanduser('~/clearml.conf'),    # used for workstation configuration (end-users, workers)
]
""" Local config files (not paths) """


LOCAL_CONFIG_FILE_OVERRIDE_VAR = EnvEntry("CLEARML_CONFIG_FILE", "TRAINS_CONFIG_FILE")
""" Local config file override environment variable. If this is set, no other local config files will be used. """


ENV_CONFIG_PATH_OVERRIDE_VAR = EnvEntry("CLEARML_CONFIG_PATH", "TRAINS_CONFIG_PATH")
"""
Environment-related config path override environment variable. If this is set, no other env config path will be used.
"""

CONFIG_VERBOSE = EnvEntry("CLEARML_CONFIG_VERBOSE", type=bool)


class Environment(object):
    """ Supported environment names """
    default = 'default'
    demo = 'demo'
    local = 'local'


CONFIG_FILE_EXTENSION = '.conf'


def is_config_file(path):
    return Path(path).suffix == CONFIG_FILE_EXTENSION


def get_active_config_file():
    f = LOCAL_CONFIG_FILE_OVERRIDE_VAR.get()
    if f and exists(expanduser(expandvars(f))):
        return f
    for f in LOCAL_CONFIG_FILES:
        if exists(expanduser(expandvars(f))):
            return f
    return None


def get_config_file():
    f = LOCAL_CONFIG_FILE_OVERRIDE_VAR.get()
    f = f if f else LOCAL_CONFIG_FILES[-1]
    return expanduser(expandvars(f)) if f else None
