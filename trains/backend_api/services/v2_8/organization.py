"""
organization service

This service provides organization level operations
"""
import six
import types
from datetime import datetime
import enum

from dateutil.parser import parse as parse_datetime

from ....backend_api.session import Request, Response, NonStrictDataModel, schema_property, StringEnum


class GetTagsRequest(Request):
    """
    Get all the user and system tags used for the company tasks and models

    :param include_system: If set to 'true' then the list of the system tags is
        also returned. The default value is 'false'
    :type include_system: bool
    :param filter: Filter on entities to collect tags from
    :type filter: dict
    """

    _service = "organization"
    _action = "get_tags"
    _version = "2.8"
    _schema = {
        'definitions': {},
        'properties': {
            'filter': {
                'description': 'Filter on entities to collect tags from',
                'properties': {
                    'system_tags': {
                        'description': "The list of system tag values to filter by. Use 'null' value to specify empty tags. Use '__Snot' value to specify that the following value should be excluded",
                        'items': {'type': 'string'},
                        'type': 'array',
                    },
                },
                'type': ['object', 'null'],
            },
            'include_system': {
                'default': False,
                'description': "If set to 'true' then the list of the system tags is also returned. The default value is 'false'",
                'type': ['boolean', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, include_system=False, filter=None, **kwargs):
        super(GetTagsRequest, self).__init__(**kwargs)
        self.include_system = include_system
        self.filter = filter

    @schema_property('include_system')
    def include_system(self):
        return self._property_include_system

    @include_system.setter
    def include_system(self, value):
        if value is None:
            self._property_include_system = None
            return
        
        self.assert_isinstance(value, "include_system", (bool,))
        self._property_include_system = value

    @schema_property('filter')
    def filter(self):
        return self._property_filter

    @filter.setter
    def filter(self, value):
        if value is None:
            self._property_filter = None
            return
        
        self.assert_isinstance(value, "filter", (dict,))
        self._property_filter = value


class GetTagsResponse(Response):
    """
    Response of organization.get_tags endpoint.

    :param tags: The list of unique tag values
    :type tags: Sequence[str]
    :param system_tags: The list of unique system tag values. Returned only if
        'include_system' is set to 'true' in the request
    :type system_tags: Sequence[str]
    """
    _service = "organization"
    _action = "get_tags"
    _version = "2.8"

    _schema = {
        'definitions': {},
        'properties': {
            'system_tags': {
                'description': "The list of unique system tag values. Returned only if 'include_system' is set to 'true' in the request",
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
            'tags': {
                'description': 'The list of unique tag values',
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, tags=None, system_tags=None, **kwargs):
        super(GetTagsResponse, self).__init__(**kwargs)
        self.tags = tags
        self.system_tags = system_tags

    @schema_property('tags')
    def tags(self):
        return self._property_tags

    @tags.setter
    def tags(self, value):
        if value is None:
            self._property_tags = None
            return
        
        self.assert_isinstance(value, "tags", (list, tuple))
        
        self.assert_isinstance(value, "tags", six.string_types, is_array=True)
        self._property_tags = value

    @schema_property('system_tags')
    def system_tags(self):
        return self._property_system_tags

    @system_tags.setter
    def system_tags(self, value):
        if value is None:
            self._property_system_tags = None
            return
        
        self.assert_isinstance(value, "system_tags", (list, tuple))
        
        self.assert_isinstance(value, "system_tags", six.string_types, is_array=True)
        self._property_system_tags = value


response_mapping = {
    GetTagsRequest: GetTagsResponse,
}
