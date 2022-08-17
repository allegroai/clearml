from __future__ import unicode_literals

import abc
import types
from argparse import Namespace
from collections import OrderedDict
from enum import Enum
from functools import reduce, wraps, WRAPPER_ASSIGNMENTS
from importlib import import_module
from itertools import chain
from operator import itemgetter
from types import ModuleType
from typing import Dict, Text, Tuple, Type, Any, Sequence

import six
from ... import services as api_services
from ....backend_api.session import CallResult
from ....backend_api.session import Session, Request as APIRequest
from ....backend_api.session.response import ResponseMeta
from ....backend_config.defs import LOCAL_CONFIG_FILE_OVERRIDE_VAR

SERVICE_TO_ENTITY_CLASS_NAMES = {"storage": "StorageItem"}


def entity_class_name(service):
    # type: (ModuleType) -> Text
    service_name = api_entity_name(service)
    return SERVICE_TO_ENTITY_CLASS_NAMES.get(service_name.lower(), service_name)


def api_entity_name(service):
    return module_name(service).rstrip("s")


@six.python_2_unicode_compatible
class APIError(Exception):
    """
    Class for representing an API error.

    self.data - ``dict`` of all returned JSON data
    self.code - HTTP response code
    self.subcode - server response subcode
    self.codes - (self.code, self.subcode) tuple
    self.message - result message sent from server
    """

    def __init__(self, response, extra_info=None):
        """
        Create a new APIError from a server response
        """
        super(APIError, self).__init__()
        self._response = response  # type: CallResult
        self.extra_info = extra_info
        self.data = response.response_data  # type: Dict
        self.meta = response.meta  # type: ResponseMeta
        self.code = response.meta.result_code  # type: int
        self.subcode = response.meta.result_subcode  # type: int
        self.message = response.meta.result_msg  # type: Text
        self.codes = (self.code, self.subcode)  # type: Tuple[int, int]

    def get_traceback(self):
        """
        Return server traceback for error, or None if doesn't exist.
        """
        try:
            return self.meta.error_stack
        except AttributeError:
            return None

    def __str__(self):
        message = "{}: ".format(type(self).__name__)
        if self.extra_info:
            message += "{}: ".format(self.extra_info)
        if not self.meta:
            message += "no meta available"
            return message
        if not self.code:
            message += "no error code available"
            return message
        message += "code {0.code}".format(self)
        if self.subcode:
            message += "/{.subcode}".format(self)
        if self.message:
            message += ": {.message}".format(self)
        return message


class StrictSession(Session):

    """
    Session that raises exceptions on errors, and be configured with explicit ``config_file`` path.
    """

    def __init__(self, config_file=None, initialize_logging=False, *args, **kwargs):
        """
        :param config_file: configuration file to use, else use the default
        :type config_file: Path | Text
        """

        def init():
            super(StrictSession, self).__init__(
                initialize_logging=initialize_logging, *args, **kwargs
            )

        if not config_file:
            init()
            return

        original = LOCAL_CONFIG_FILE_OVERRIDE_VAR.get() or None
        try:
            LOCAL_CONFIG_FILE_OVERRIDE_VAR.set(str(config_file))
            init()
        finally:
            if original is None:
                LOCAL_CONFIG_FILE_OVERRIDE_VAR.pop()
            else:
                LOCAL_CONFIG_FILE_OVERRIDE_VAR.set(original)

    def send(self, request, *args, **kwargs):
        result = super(StrictSession, self).send(request, *args, **kwargs)
        if not result.ok():
            raise APIError(result)
        if not result.response:
            raise APIError(result, extra_info="Invalid response")
        return result


class Response(object):

    """
    Proxy object for API result data.
    Exposes "meta" of the original result.
    """

    def __init__(self, result, dest=None):
        """
        :param result: result of endpoint call
        :type result: CallResult
        :param dest: if all of a response's data is contained in one field, use that field
        :type dest: Text
        """
        self.response = None
        self._result = result
        response = getattr(result, "response", result)
        if getattr(response, "_service") == "events" and \
                getattr(response, "_action") in ("scalar_metrics_iter_histogram",
                                                 "multi_task_scalar_metrics_iter_histogram",
                                                 "vector_metrics_iter_histogram",
                                                 ):
            # put all the response data under metrics:
            response.metrics = result.response_data
            # noinspection PyProtectedMember
            if 'metrics' not in response.__class__._get_data_props():
                # noinspection PyProtectedMember
                response.__class__._data_props_list['metrics'] = 'metrics'
        if dest:
            response = getattr(response, dest)
        self.response = response

    def __getattr__(self, attr):
        if self.response is None:
            return None
        return getattr(self.response, attr)

    @property
    def meta(self):
        return self._result.meta

    def __repr__(self):
        return repr(self.response)

    def __dir__(self):
        fields = [
            name
            for name in dir(self.response)
            if isinstance(getattr(type(self.response), name, None), property)
        ]
        return list(set(chain(super(Response, self).__dir__(), fields)) - {"response"})


@six.python_2_unicode_compatible
class TableResponse(Response):

    """
    Representation of result containing an array of entities
    """

    def __init__(
        self,
        service,  # type: Service
        entity,  # type: Type[entity]
        fields=None,  # type: Sequence[Text]
        *args,
        **kwargs
    ):
        """
        :param service: service of entity
        :param entity: class representing entity
        :param fields: entity attributes requested by client
        """
        super(TableResponse, self).__init__(*args, **kwargs)
        self.service = service
        self.entity = entity
        self.fields = fields or ("id", "name")
        self.response = [entity(service, item) for item in self]

    def __repr__(self, fields=None):
        return self._format_table(fields=fields)

    __str__ = __repr__

    def _format_table(self, fields=None):
        """
        Display <fields> attributes of each element in a table
        :param fields:
        """

        def getter(obj, attr):
            result = reduce(
                lambda x, name: x if x is None else getattr(x, name, None),
                attr.split("."),
                obj,
            )
            return "" if result is None else result

        fields = fields or self.fields
        return '\n'.join(str(dict((attr, getter(item, attr)) for attr in fields)) for item in self)

    def display(self, fields=None):
        print(self._format_table(fields=fields))

    def where(self, predicate=None, **kwargs):
        """
        Filter items.
        <predicate> is a callable from a single item to a boolean. Items for which <predicate> is True will be returned.
        Keyword arguments are interpreted as attribute equivalence, meaning:
        tasks.where(name='foo')
        will return only datasets whose name is "foo".

        Giving more than one condition (predicate and keyword arguments) establishes an "and" relation.
        """

        def compare_enum(x, y):
            return x == y or isinstance(x, Enum) and x.value == y

        return TableResponse(
            self.service,
            self.entity,
            self.fields,
            [
                item
                for item in self
                if (not predicate or predicate(item))
                and all(
                    compare_enum(getattr(item, key), value)
                    for key, value in kwargs.items()
                )
            ],
        )

    def __getitem__(self, item):
        return self.response[item]

    def __iter__(self):
        return iter(self.response)

    def __len__(self):
        return len(self.response)


@six.add_metaclass(abc.ABCMeta)
class Entity(object):

    """
    Represent a server object.
    Enables calls like:
    >>> client = APIClient()
    >>> entity = client.service.get_by_id(entity_id)
    >>> entity.action(**kwargs)
    instead of:
    >>> client.service.action(id=entity_id, **kwargs)
    """

    @property
    @abc.abstractmethod
    def entity_name(self):  # type: () -> Text
        """
        Singular name of entity
        """
        pass

    @property
    @abc.abstractmethod
    def get_by_id_request(self):  # type: () -> Type[APIRequest]
        """
        get_by_id request class
        """
        pass

    def __init__(self, service, data):
        self._service = service
        self.data = getattr(data, self.entity_name, data)
        self.__doc__ = self.data.__doc__

    def fetch(self):
        """
        Update the entity data from the server.
        """
        result = self._service.session.send(self.get_by_id_request(self.data.id))
        self.data = getattr(result.response, self.entity_name)

    def _get_default_kwargs(self):
        return {self.entity_name: self.data.id}

    def __getattr__(self, attr):
        """
        Inject the entity's ID to the method call.
        All missing properties are assumed to be functions.
        """
        try:
            return getattr(self.data, attr)
        except AttributeError:
            pass

        func = getattr(self._service, attr)
        if not callable(func):
            return func

        @wrap_request_class(func)
        def new_func(*args, **kwargs):
            kwargs = dict(self._get_default_kwargs(), **kwargs)
            return func(*args, **kwargs)

        return new_func

    def __dir__(self):
        """
        Add ``self._service``'s methods to ``dir`` result.
        """
        try:
            dir_ = super(Entity, self).__dir__
        except AttributeError:
            base = self.__dict__
        else:
            base = dir_()
        return list(set(base).union(dir(self._service), dir(self.data)))

    def __repr__(self):
        """
        Display entity type, ID, and - if available - name.
        """
        parts = (type(self).__name__, ": ", "id={}".format(self.data.id))
        try:
            parts += (", ", 'name="{}"'.format(self.data.name))
        except AttributeError:
            pass
        return "<{}>".format("".join(parts))


def wrap_request_class(cls):
    return wraps(cls, assigned=tuple(WRAPPER_ASSIGNMENTS) + ("from_dict",))


def make_action(service, request_cls):
    # noinspection PyProtectedMember
    action = request_cls._action
    try:
        get_by_id_request = service.GetByIdRequest
    except AttributeError:
        get_by_id_request = None

    wrap = wrap_request_class(request_cls)

    if action not in ["get_all", "get_all_ex", "get_by_id", "create"]:

        @wrap
        def new_func(self, *args, **kwargs):
            return Response(self.session.send(request_cls(*args, **kwargs)))

        new_func.__name__ = new_func.__qualname__ = action
        return new_func

    entity_name = api_entity_name(service)
    class_name = entity_class_name(service).capitalize()
    properties = {
        "__module__": __name__,
        "entity_name": entity_name.lower(),
        "get_by_id_request": get_by_id_request,
    }
    entity = type(str(class_name), (Entity,), properties)

    if action == "get_by_id":

        @wrap
        def get(self, *args, **kwargs):
            return entity(
                self, self.session.send(request_cls(*args, **kwargs)).response
            )

    elif action == "create":

        @wrap
        def get(self, *args, **kwargs):
            return entity(
                self,
                Namespace(
                    id=self.session.send(request_cls(*args, **kwargs)).response.id
                ),
            )

    elif action in ["get_all", "get_all_ex"]:
        # noinspection PyProtectedMember
        for dest in service.response_mapping[request_cls]._get_data_props().keys():
            if dest != "scroll_id":
                break

        @wrap
        def get(self, *args, **kwargs):
            return TableResponse(
                service=self,
                entity=entity,
                result=self.session.send(request_cls(*args, **kwargs)),
                dest=dest,
                fields=kwargs.pop("only_fields", None),
            )

    else:
        assert False

    get.__name__ = get.__qualname__ = action

    return get


@six.add_metaclass(abc.ABCMeta)
class Service(object):

    """
    Superclass for action-grouping classes.
    """

    name = abc.abstractproperty()
    __doc__ = abc.abstractproperty()

    def __init__(self, session):
        self.session = session


def get_requests(service):
    # force load proxy object
    # noinspection PyBroadException
    try:
        service.dummy
    except Exception:
        pass

    # noinspection PyProtectedMember
    return OrderedDict(
        (key, value)
        for key, value in sorted(
            vars(service.__wrapped__ if hasattr(service, '__wrapped__') else service).items(), key=itemgetter(0))
        if isinstance(value, type) and issubclass(value, APIRequest) and value._action
    )


def make_service_class(module):
    # type: (...) -> Type[Service]
    """
    Create a service class from service module.
    """
    properties = OrderedDict(
        [
            ("__module__", __name__),
            ("__doc__", module.__doc__),
            ("name", module_name(module)),
        ]
    )
    properties.update(
        (f.__name__, f)
        for f in (
            make_action(module, value) for key, value in get_requests(module).items()
        )
    )
    # noinspection PyTypeChecker
    return type(str(module_name(module)), (Service,), properties)


def module_name(module):
    try:
        module = module.__name__
    except AttributeError:
        pass
    base_name = module.split(".")[-1]
    return "".join(s.capitalize() for s in base_name.split("_"))


class Version(Entity):
    entity_name = "version"
    get_by_id_request = None

    def fetch(self):
        try:
            published = self.data.status == "published"
        except AttributeError:
            published = False

        self.data = self._service.get_versions(
            dataset=self.dataset, only_published=published, versions=[self.id]
        )[0].data

    def _get_default_kwargs(self):
        return dict(
            super(Version, self)._get_default_kwargs(), **{"dataset": self.data.dataset}
        )


class APIClient(object):

    auth = None  # type: Any
    queues = None  # type: Any
    tasks = None  # type: Any
    workers = None  # type: Any
    events = None  # type: Any
    models = None  # type: Any
    projects = None  # type: Any

    def __init__(self, session=None, api_version=None, **kwargs):
        self.session = session or StrictSession()

        _api_services = kwargs.pop("api_services", api_services)

        def import_(*args, **kwargs):
            try:
                return import_module(*args, **kwargs)
            except ImportError:
                return None

        if api_version:
            api_version = "v{}".format(str(api_version).replace(".", "_"))
            services = OrderedDict(
                (name, mod)
                for name, mod in (
                    (
                        name,
                        import_(".".join((_api_services.__name__, api_version, name))),
                    )
                    for name in _api_services.__all__
                )
                if mod
            )
        else:
            services = OrderedDict(
                (name, getattr(_api_services, name)) for name in _api_services.__all__
            )
        self._update_services(services)

    def _update_services(self, services):
        # type: (Dict[str, types.ModuleType]) -> ()
        self.__dict__.update(
            dict(
                {
                    name: make_service_class(module)(self.session)
                    for name, module in services.items()
                },
            )
        )
