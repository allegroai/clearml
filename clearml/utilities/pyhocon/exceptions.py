class ConfigException(Exception):

    def __init__(self, message, ex=None):
        super(ConfigException, self).__init__(message)
        self._exception = ex


class ConfigMissingException(ConfigException, KeyError):
    pass


class ConfigSubstitutionException(ConfigException):
    pass


class ConfigWrongTypeException(ConfigException):
    pass
