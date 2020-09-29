import requests

import six

from . import jsonmodels
from .apimodel import ApiModel
from .datamodel import NonStrictDataModelMixin


class FloatOrStringField(jsonmodels.fields.BaseField):

    """String field."""

    types = (float, six.string_types,)


class Response(ApiModel, NonStrictDataModelMixin):
    pass


class _ResponseEndpoint(jsonmodels.models.Base):
    name = jsonmodels.fields.StringField()
    requested_version = FloatOrStringField()
    actual_version = FloatOrStringField()


class ResponseMeta(jsonmodels.models.Base):
    @property
    def is_valid(self):
        return self._is_valid

    @classmethod
    def from_raw_data(cls, status_code, text="", endpoint=None):
        return cls(is_valid=False, result_code=status_code, result_subcode=0, result_msg=text,
                   endpoint=_ResponseEndpoint(name=(endpoint or 'unknown')))

    def __init__(self, is_valid=True, **kwargs):
        super(ResponseMeta, self).__init__(**kwargs)
        self._is_valid = is_valid

    id = jsonmodels.fields.StringField(required=True)
    trx = jsonmodels.fields.StringField(required=True)
    endpoint = jsonmodels.fields.EmbeddedField([_ResponseEndpoint], required=True)
    result_code = jsonmodels.fields.IntField(required=True)
    result_subcode = jsonmodels.fields.IntField()
    result_msg = jsonmodels.fields.StringField(required=True)
    error_stack = jsonmodels.fields.StringField()

    def __str__(self):
        if self.result_code == requests.codes.ok:
            return "<%d: %s/v%s>" % (self.result_code, self.endpoint.name, self.endpoint.actual_version)
        elif self._is_valid:
            return "<%d/%d: %s/v%s (%s)>" % (self.result_code, self.result_subcode, self.endpoint.name,
                                             self.endpoint.actual_version, self.result_msg)
        return "<%d/%d: %s (%s)>" % (self.result_code, self.result_subcode, self.endpoint.name, self.result_msg)
