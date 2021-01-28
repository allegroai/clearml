class ConfigurationError(Exception):

    def __init__(self, msg, file_path=None, *args):
        super(ConfigurationError, self).__init__(msg, *args)
        self.file_path = file_path
