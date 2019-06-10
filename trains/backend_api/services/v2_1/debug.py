"""
debug service

Debugging utilities
"""
import six
import types
from datetime import datetime
import enum

from dateutil.parser import parse as parse_datetime

from ....backend_api.session import Request, BatchRequest, Response, DataModel, NonStrictDataModel, CompoundRequest, schema_property, StringEnum


class ApiexRequest(Request):
    """
    """

    _service = "debug"
    _action = "apiex"
    _version = "1.5"
    _schema = {'definitions': {}, 'properties': {}, 'required': [], 'type': 'object'}


class ApiexResponse(Response):
    """
    Response of debug.apiex endpoint.

    """
    _service = "debug"
    _action = "apiex"
    _version = "1.5"

    _schema = {'definitions': {}, 'properties': {}, 'type': 'object'}


class EchoRequest(Request):
    """
    Return request data

    """

    _service = "debug"
    _action = "echo"
    _version = "1.5"
    _schema = {'definitions': {}, 'properties': {}, 'type': 'object'}


class EchoResponse(Response):
    """
    Response of debug.echo endpoint.

    """
    _service = "debug"
    _action = "echo"
    _version = "1.5"

    _schema = {'definitions': {}, 'properties': {}, 'type': 'object'}


class ExRequest(Request):
    """
    """

    _service = "debug"
    _action = "ex"
    _version = "1.5"
    _schema = {'definitions': {}, 'properties': {}, 'required': [], 'type': 'object'}


class ExResponse(Response):
    """
    Response of debug.ex endpoint.

    """
    _service = "debug"
    _action = "ex"
    _version = "1.5"

    _schema = {'definitions': {}, 'properties': {}, 'type': 'object'}


class PingRequest(Request):
    """
    Return a message. Does not require authorization.

    """

    _service = "debug"
    _action = "ping"
    _version = "1.5"
    _schema = {'definitions': {}, 'properties': {}, 'type': 'object'}


class PingResponse(Response):
    """
    Response of debug.ping endpoint.

    :param msg: A friendly message
    :type msg: str
    """
    _service = "debug"
    _action = "ping"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'msg': {
                'description': 'A friendly message',
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, msg=None, **kwargs):
        super(PingResponse, self).__init__(**kwargs)
        self.msg = msg

    @schema_property('msg')
    def msg(self):
        return self._property_msg

    @msg.setter
    def msg(self, value):
        if value is None:
            self._property_msg = None
            return
        
        self.assert_isinstance(value, "msg", six.string_types)
        self._property_msg = value


class PingAuthRequest(Request):
    """
    Return a message. Requires authorization.

    """

    _service = "debug"
    _action = "ping_auth"
    _version = "1.5"
    _schema = {'definitions': {}, 'properties': {}, 'type': 'object'}


class PingAuthResponse(Response):
    """
    Response of debug.ping_auth endpoint.

    :param msg: A friendly message
    :type msg: str
    """
    _service = "debug"
    _action = "ping_auth"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'msg': {
                'description': 'A friendly message',
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, msg=None, **kwargs):
        super(PingAuthResponse, self).__init__(**kwargs)
        self.msg = msg

    @schema_property('msg')
    def msg(self):
        return self._property_msg

    @msg.setter
    def msg(self, value):
        if value is None:
            self._property_msg = None
            return
        
        self.assert_isinstance(value, "msg", six.string_types)
        self._property_msg = value


response_mapping = {
    EchoRequest: EchoResponse,
    PingRequest: PingResponse,
    PingAuthRequest: PingAuthResponse,
    ApiexRequest: ApiexResponse,
    ExRequest: ExResponse,
}
