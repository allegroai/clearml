import abc

import jsonschema
import six

from .apimodel import ApiModel
from .datamodel import DataModel


class Request(ApiModel):
    _method = 'get'

    def __init__(self, **kwargs):
        if kwargs:
            raise ValueError('Unsupported keyword arguments: %s' % ', '.join(kwargs.keys()))


@six.add_metaclass(abc.ABCMeta)
class BatchRequest(Request):

    _batched_request_cls = abc.abstractproperty()

    _schema_errors = (jsonschema.SchemaError, jsonschema.ValidationError, jsonschema.FormatError,
                      jsonschema.RefResolutionError)

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
            except (jsonschema.SchemaError, jsonschema.ValidationError,
                    jsonschema.FormatError, jsonschema.RefResolutionError) as e:
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
        return self._get_item().to_dict()

    def validate(self):
        return self._get_item().validate(self._schema)
