"""
news service

This service provides platform news.
"""
import six
import types
from datetime import datetime
import enum

from dateutil.parser import parse as parse_datetime

from ....backend_api.session import Request, BatchRequest, Response, DataModel, NonStrictDataModel, CompoundRequest, schema_property, StringEnum


class GetRequest(Request):
    """
    Gets latest news link

    """

    _service = "news"
    _action = "get"
    _version = "1.5"
    _schema = {'definitions': {}, 'properties': {}, 'type': 'object'}


class GetResponse(Response):
    """
    Response of news.get endpoint.

    :param url: URL to news html file
    :type url: str
    """
    _service = "news"
    _action = "get"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'url': {
                'description': 'URL to news html file',
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, url=None, **kwargs):
        super(GetResponse, self).__init__(**kwargs)
        self.url = url

    @schema_property('url')
    def url(self):
        return self._property_url

    @url.setter
    def url(self, value):
        if value is None:
            self._property_url = None
            return
        
        self.assert_isinstance(value, "url", six.string_types)
        self._property_url = value


response_mapping = {
    GetRequest: GetResponse,
}
