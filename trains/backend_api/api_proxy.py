from .session import Session
import importlib


class ApiServiceProxy(object):
    def __init__(self, module):
        self.__wrapped_name__ = module
        self.__wrapped_version__ = Session.api_version

    def __getattr__(self, attr):
        if attr in ['__wrapped_name__', '__wrapped__', '__wrapped_version__']:
            return self.__dict__.get(attr)

        if not self.__dict__.get('__wrapped__') or self.__dict__.get('__wrapped_version__') != Session.api_version:
            self.__dict__['__wrapped_version__'] = Session.api_version
            self.__dict__['__wrapped__'] = importlib.import_module('.v'+str(Session.api_version).replace('.', '_') +
                                                                   '.' + self.__dict__.get('__wrapped_name__'),
                                                                   package='trains.backend_api.services')
        return getattr(self.__dict__['__wrapped__'], attr)
