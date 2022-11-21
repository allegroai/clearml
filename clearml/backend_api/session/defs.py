from ...backend_config import EnvEntry
from ...backend_config.converters import safe_text_to_bool


ENV_HOST = EnvEntry("CLEARML_API_HOST", "TRAINS_API_HOST")
ENV_WEB_HOST = EnvEntry("CLEARML_WEB_HOST", "TRAINS_WEB_HOST")
ENV_FILES_HOST = EnvEntry("CLEARML_FILES_HOST", "TRAINS_FILES_HOST")
ENV_ACCESS_KEY = EnvEntry("CLEARML_API_ACCESS_KEY", "TRAINS_API_ACCESS_KEY")
ENV_SECRET_KEY = EnvEntry("CLEARML_API_SECRET_KEY", "TRAINS_API_SECRET_KEY")
ENV_AUTH_TOKEN = EnvEntry("CLEARML_AUTH_TOKEN")
ENV_VERBOSE = EnvEntry(
    "CLEARML_API_VERBOSE", "TRAINS_API_VERBOSE", converter=safe_text_to_bool, type=bool, default=False
)
ENV_HOST_VERIFY_CERT = EnvEntry("CLEARML_API_HOST_VERIFY_CERT", "TRAINS_API_HOST_VERIFY_CERT",
                                type=bool, default=True)
ENV_OFFLINE_MODE = EnvEntry("CLEARML_OFFLINE_MODE", "TRAINS_OFFLINE_MODE", type=bool, converter=safe_text_to_bool)
ENV_CLEARML_NO_DEFAULT_SERVER = EnvEntry(
    "CLEARML_NO_DEFAULT_SERVER", "TRAINS_NO_DEFAULT_SERVER", converter=safe_text_to_bool, type=bool, default=True
)
ENV_DISABLE_VAULT_SUPPORT = EnvEntry('CLEARML_DISABLE_VAULT_SUPPORT', type=bool)
ENV_ENABLE_ENV_CONFIG_SECTION = EnvEntry('CLEARML_ENABLE_ENV_CONFIG_SECTION', type=bool)
ENV_ENABLE_FILES_CONFIG_SECTION = EnvEntry('CLEARML_ENABLE_FILES_CONFIG_SECTION', type=bool)
ENV_DEFERRED_TASK_INIT = EnvEntry('CLEARML_DEFERRED_TASK_INIT', type=bool)
ENV_IGNORE_MISSING_CONFIG = EnvEntry("CLEARML_IGNORE_MISSING_CONFIG", type=bool)

"""
Experimental option to set the request method for all API requests and auth login.
This could be useful when GET requests with payloads are blocked by a server as
POST requests can be used instead.

However this has not been vigorously tested and may have unintended consequences.
"""
ENV_API_DEFAULT_REQ_METHOD = EnvEntry("CLEARML_API_DEFAULT_REQ_METHOD")

"""
Experimental option to make the SDK retry on additional error codes.
Use a comma-separated list of integer return codes.
NOTE: this changes behavior and might cause the experiment to wait
for a very long time for a non-responding or mis-configured server
"""
ENV_API_EXTRA_RETRY_CODES = EnvEntry("CLEARML_API_EXTRA_RETRY_CODES")


class MissingConfigError(ValueError):
    def __init__(self, message=None):
        if message is None:
            message = (
                "It seems ClearML is not configured on this machine!\n"
                "To get started with ClearML, setup your own 'clearml-server' or "
                "create a free account at https://app.clear.ml\n"
                "Setup instructions can be found here: https://clear.ml/docs"
            )
        super(MissingConfigError, self).__init__(message)
