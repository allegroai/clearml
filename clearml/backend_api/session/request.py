import abc

import six
from jsonschema.exceptions import FormatError, SchemaError, ValidationError

try:
    # Since `referencing`` only supports Python >= 3.8, this try-except blocks maintain support
    # for earlier python versions.
    from referencing.exceptions import Unresolvable
except ImportError:
    from jsonschema.exceptions import RefResolutionError as Unresolvable

from .apimodel import ApiModel
from .datamodel import DataModel
from .defs import ENV_API_DEFAULT_REQ_METHOD


if ENV_API_DEFAULT_REQ_METHOD.exists() and ENV_API_DEFAULT_REQ_METHOD.get().upper() not in ("GET", "POST", "PUT"):
    raise ValueError(
        "CLEARML_API_DEFAULT_REQ_METHOD environment variable must be 'get' or 'post' (any case is allowed)."
    )


class Request(ApiModel):
    def_method = ENV_API_DEFAULT_REQ_METHOD.get(default="get")
    _method = ENV_API_DEFAULT_REQ_METHOD.get(default="get")

    def __init__(self, **kwargs):
        allow_extra_fields = kwargs.pop("_allow_extra_fields_", False)
        if not allow_extra_fields and kwargs:
            raise ValueError('Unsupported keyword arguments: %s' % ', '.join(kwargs.keys()))
        elif allow_extra_fields and kwargs:
            self._extra_fields = kwargs
        else:
            self._extra_fields = {}

    def to_dict(self, *args, **kwargs):
        res = super(Request, self).to_dict(*args, **kwargs)
        if self._extra_fields:
            res.update(self._extra_fields)
        return res


@six.add_metaclass(abc.ABCMeta)
class BatchRequest(Request):

    _batched_request_cls = abc.abstractproperty()

    _schema_errors = (SchemaError, ValidationError, FormatError, Unresolvable)

    def __init__(self, requests, validate_requests=False, allow_raw_requests=True, **kwargs):
        super(BatchRequest, self).__init__(**kwargs)
        self._validate_requests = validate_requests
        self._allow_raw_requests = allow_raw_requests
        self._property_requests = None
        self.requests = requests

    @property
    def requests(self):
        return self._property_requests

    @requests.setter
    def requests(self, value):
        assert issubclass(self._batched_request_cls, Request)
        assert isinstance(value, (list, tuple))
        if not self._allow_raw_requests:
            if any(isinstance(x, dict) for x in value):
                value = [self._batched_request_cls(**x) if isinstance(x, dict) else x for x in value]
            assert all(isinstance(x, self._batched_request_cls) for x in value)

        self._property_requests = value

    def validate(self):
        if not self._validate_requests or self._allow_raw_requests:
            return
        for i, req in enumerate(self.requests):
            try:
                req.validate()
            except (SchemaError, ValidationError, FormatError, Unresolvable) as e:
                raise Exception('Validation error in batch item #%d: %s' % (i, str(e)))

    def get_json(self):
        return [r if isinstance(r, dict) else r.to_dict() for r in self.requests]


class CompoundRequest(Request):
    _item_prop_name = 'item'

    def _get_item(self):
        item = getattr(self, self._item_prop_name, None)
        if item is None:
            raise ValueError('Item property is empty or missing')
        assert isinstance(item, DataModel)
        return item

    def to_dict(self):
        dict_ = self._get_item().to_dict()
        dict_properties = super(Request, self).to_dict()
        if self._item_prop_name in dict_properties:
            del dict_properties[self._item_prop_name]
        dict_.update(dict_properties)
        return dict_

    def validate(self):
        return self._get_item().validate(self._schema)
