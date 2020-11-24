from ...backend_config import EnvEntry
from ...backend_config.converters import safe_text_to_bool


ENV_HOST = EnvEntry("TRAINS_API_HOST", "ALG_API_HOST")
ENV_WEB_HOST = EnvEntry("TRAINS_WEB_HOST", "ALG_WEB_HOST")
ENV_FILES_HOST = EnvEntry("TRAINS_FILES_HOST", "ALG_FILES_HOST")
ENV_ACCESS_KEY = EnvEntry("TRAINS_API_ACCESS_KEY", "ALG_API_ACCESS_KEY")
ENV_SECRET_KEY = EnvEntry("TRAINS_API_SECRET_KEY", "ALG_API_SECRET_KEY")
ENV_VERBOSE = EnvEntry("TRAINS_API_VERBOSE", "ALG_API_VERBOSE", type=bool, default=False)
ENV_HOST_VERIFY_CERT = EnvEntry("TRAINS_API_HOST_VERIFY_CERT", "ALG_API_HOST_VERIFY_CERT", type=bool, default=True)
ENV_OFFLINE_MODE = EnvEntry("TRAINS_OFFLINE_MODE", "ALG_OFFLINE_MODE", type=bool, converter=safe_text_to_bool)
ENV_TRAINS_NO_DEFAULT_SERVER = EnvEntry("TRAINS_NO_DEFAULT_SERVER", "ALG_NO_DEFAULT_SERVER", type=bool, default=False)
