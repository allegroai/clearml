"""
storage service

Provides a management API for customer-associated storage locations
"""
import six
import types
from datetime import datetime
import enum

from dateutil.parser import parse as parse_datetime

from ....backend_api.session import Request, BatchRequest, Response, DataModel, NonStrictDataModel, CompoundRequest, schema_property, StringEnum


class Credentials(NonStrictDataModel):
    """
    :param access_key: Credentials access key
    :type access_key: str
    :param secret_key: Credentials secret key
    :type secret_key: str
    """
    _schema = {
        'properties': {
            'access_key': {
                'description': 'Credentials access key',
                'type': ['string', 'null'],
            },
            'secret_key': {
                'description': 'Credentials secret key',
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, access_key=None, secret_key=None, **kwargs):
        super(Credentials, self).__init__(**kwargs)
        self.access_key = access_key
        self.secret_key = secret_key

    @schema_property('access_key')
    def access_key(self):
        return self._property_access_key

    @access_key.setter
    def access_key(self, value):
        if value is None:
            self._property_access_key = None
            return
        
        self.assert_isinstance(value, "access_key", six.string_types)
        self._property_access_key = value

    @schema_property('secret_key')
    def secret_key(self):
        return self._property_secret_key

    @secret_key.setter
    def secret_key(self, value):
        if value is None:
            self._property_secret_key = None
            return
        
        self.assert_isinstance(value, "secret_key", six.string_types)
        self._property_secret_key = value


class Storage(NonStrictDataModel):
    """
    :param id: Entry ID
    :type id: str
    :param name: Entry name
    :type name: str
    :param company: Company ID
    :type company: str
    :param created: Entry creation time
    :type created: datetime.datetime
    :param uri: Storage URI
    :type uri: str
    :param credentials: Credentials required for accessing the storage
    :type credentials: Credentials
    """
    _schema = {
        'properties': {
            'company': {'description': 'Company ID', 'type': ['string', 'null']},
            'created': {
                'description': 'Entry creation time',
                'format': 'date-time',
                'type': ['string', 'null'],
            },
            'credentials': {
                'description': 'Credentials required for accessing the storage',
                'oneOf': [{'$ref': '#/definitions/credentials'}, {'type': 'null'}],
            },
            'id': {'description': 'Entry ID', 'type': ['string', 'null']},
            'name': {'description': 'Entry name', 'type': ['string', 'null']},
            'uri': {'description': 'Storage URI', 'type': ['string', 'null']},
        },
        'type': 'object',
    }
    def __init__(
            self, id=None, name=None, company=None, created=None, uri=None, credentials=None, **kwargs):
        super(Storage, self).__init__(**kwargs)
        self.id = id
        self.name = name
        self.company = company
        self.created = created
        self.uri = uri
        self.credentials = credentials

    @schema_property('id')
    def id(self):
        return self._property_id

    @id.setter
    def id(self, value):
        if value is None:
            self._property_id = None
            return
        
        self.assert_isinstance(value, "id", six.string_types)
        self._property_id = value

    @schema_property('name')
    def name(self):
        return self._property_name

    @name.setter
    def name(self, value):
        if value is None:
            self._property_name = None
            return
        
        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property('company')
    def company(self):
        return self._property_company

    @company.setter
    def company(self, value):
        if value is None:
            self._property_company = None
            return
        
        self.assert_isinstance(value, "company", six.string_types)
        self._property_company = value

    @schema_property('created')
    def created(self):
        return self._property_created

    @created.setter
    def created(self, value):
        if value is None:
            self._property_created = None
            return
        
        self.assert_isinstance(value, "created", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_created = value

    @schema_property('uri')
    def uri(self):
        return self._property_uri

    @uri.setter
    def uri(self, value):
        if value is None:
            self._property_uri = None
            return
        
        self.assert_isinstance(value, "uri", six.string_types)
        self._property_uri = value

    @schema_property('credentials')
    def credentials(self):
        return self._property_credentials

    @credentials.setter
    def credentials(self, value):
        if value is None:
            self._property_credentials = None
            return
        if isinstance(value, dict):
            value = Credentials.from_dict(value)
        else:
            self.assert_isinstance(value, "credentials", Credentials)
        self._property_credentials = value


class CreateRequest(Request):
    """
    Create a new storage entry

    :param name: Storage name
    :type name: str
    :param uri: Storage URI
    :type uri: str
    :param credentials: Credentials required for accessing the storage
    :type credentials: Credentials
    :param company: Company under which to add this storage. Only valid for users
        with the root or system role, otherwise the calling user's company will be
        used.
    :type company: str
    """

    _service = "storage"
    _action = "create"
    _version = "1.5"
    _schema = {
        'definitions': {
            'credentials': {
                'properties': {
                    'access_key': {
                        'description': 'Credentials access key',
                        'type': ['string', 'null'],
                    },
                    'secret_key': {
                        'description': 'Credentials secret key',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
        },
        'properties': {
            'company': {
                'description': "Company under which to add this storage. Only valid for users with the root or system role, otherwise the calling user's company will be used.",
                'type': 'string',
            },
            'credentials': {
                '$ref': '#/definitions/credentials',
                'description': 'Credentials required for accessing the storage',
            },
            'name': {'description': 'Storage name', 'type': ['string', 'null']},
            'uri': {'description': 'Storage URI', 'type': 'string'},
        },
        'required': ['uri'],
        'type': 'object',
    }
    def __init__(
            self, uri, name=None, credentials=None, company=None, **kwargs):
        super(CreateRequest, self).__init__(**kwargs)
        self.name = name
        self.uri = uri
        self.credentials = credentials
        self.company = company

    @schema_property('name')
    def name(self):
        return self._property_name

    @name.setter
    def name(self, value):
        if value is None:
            self._property_name = None
            return
        
        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property('uri')
    def uri(self):
        return self._property_uri

    @uri.setter
    def uri(self, value):
        if value is None:
            self._property_uri = None
            return
        
        self.assert_isinstance(value, "uri", six.string_types)
        self._property_uri = value

    @schema_property('credentials')
    def credentials(self):
        return self._property_credentials

    @credentials.setter
    def credentials(self, value):
        if value is None:
            self._property_credentials = None
            return
        if isinstance(value, dict):
            value = Credentials.from_dict(value)
        else:
            self.assert_isinstance(value, "credentials", Credentials)
        self._property_credentials = value

    @schema_property('company')
    def company(self):
        return self._property_company

    @company.setter
    def company(self, value):
        if value is None:
            self._property_company = None
            return
        
        self.assert_isinstance(value, "company", six.string_types)
        self._property_company = value


class CreateResponse(Response):
    """
    Response of storage.create endpoint.

    :param id: New storage ID
    :type id: str
    """
    _service = "storage"
    _action = "create"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'id': {'description': 'New storage ID', 'type': ['string', 'null']},
        },
        'type': 'object',
    }
    def __init__(
            self, id=None, **kwargs):
        super(CreateResponse, self).__init__(**kwargs)
        self.id = id

    @schema_property('id')
    def id(self):
        return self._property_id

    @id.setter
    def id(self, value):
        if value is None:
            self._property_id = None
            return
        
        self.assert_isinstance(value, "id", six.string_types)
        self._property_id = value


class DeleteRequest(Request):
    """
    Deletes a storage entry

    :param storage: Storage entry ID
    :type storage: str
    """

    _service = "storage"
    _action = "delete"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'storage': {'description': 'Storage entry ID', 'type': 'string'},
        },
        'required': ['storage'],
        'type': 'object',
    }
    def __init__(
            self, storage, **kwargs):
        super(DeleteRequest, self).__init__(**kwargs)
        self.storage = storage

    @schema_property('storage')
    def storage(self):
        return self._property_storage

    @storage.setter
    def storage(self, value):
        if value is None:
            self._property_storage = None
            return
        
        self.assert_isinstance(value, "storage", six.string_types)
        self._property_storage = value


class DeleteResponse(Response):
    """
    Response of storage.delete endpoint.

    :param deleted: Number of storage entries deleted (0 or 1)
    :type deleted: int
    """
    _service = "storage"
    _action = "delete"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'deleted': {
                'description': 'Number of storage entries deleted (0 or 1)',
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, deleted=None, **kwargs):
        super(DeleteResponse, self).__init__(**kwargs)
        self.deleted = deleted

    @schema_property('deleted')
    def deleted(self):
        return self._property_deleted

    @deleted.setter
    def deleted(self, value):
        if value is None:
            self._property_deleted = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "deleted", six.integer_types)
        self._property_deleted = value


class GetAllRequest(Request):
    """
    Get all storage entries

    :param name: Get only storage entries whose name matches this pattern (python
        regular expression syntax)
    :type name: str
    :param id: List of Storage IDs used to filter results
    :type id: Sequence[str]
    :param page: Page number, returns a specific page out of the result list of
        results.
    :type page: int
    :param page_size: Page size, specifies the number of results returned in each
        page (last page may contain fewer results)
    :type page_size: int
    :param order_by: List of field names to order by. When search_text is used,
        '@text_score' can be used as a field representing the text score of returned
        documents. Use '-' prefix to specify descending order. Optional, recommended
        when using page
    :type order_by: Sequence[str]
    :param only_fields: List of document field names (nesting is supported using
        '.', e.g. execution.model_labels). If provided, this list defines the query's
        projection (only these fields will be returned for each result entry)
    :type only_fields: Sequence[str]
    """

    _service = "storage"
    _action = "get_all"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'id': {
                'description': 'List of Storage IDs used to filter results',
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
            'name': {
                'description': 'Get only storage entries whose name matches this pattern (python regular expression syntax)',
                'type': ['string', 'null'],
            },
            'only_fields': {
                'description': "List of document field names (nesting is supported using '.', e.g. execution.model_labels). If provided, this list defines the query's projection (only these fields will be returned for each result entry)",
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
            'order_by': {
                'description': "List of field names to order by. When search_text is used, '@text_score' can be used as a field representing the text score of returned documents. Use '-' prefix to specify descending order. Optional, recommended when using page",
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
            'page': {
                'description': 'Page number, returns a specific page out of the result list of results.',
                'minimum': 0,
                'type': ['integer', 'null'],
            },
            'page_size': {
                'description': 'Page size, specifies the number of results returned in each page (last page may contain fewer results)',
                'minimum': 1,
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, name=None, id=None, page=None, page_size=None, order_by=None, only_fields=None, **kwargs):
        super(GetAllRequest, self).__init__(**kwargs)
        self.name = name
        self.id = id
        self.page = page
        self.page_size = page_size
        self.order_by = order_by
        self.only_fields = only_fields

    @schema_property('name')
    def name(self):
        return self._property_name

    @name.setter
    def name(self, value):
        if value is None:
            self._property_name = None
            return
        
        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property('id')
    def id(self):
        return self._property_id

    @id.setter
    def id(self, value):
        if value is None:
            self._property_id = None
            return
        
        self.assert_isinstance(value, "id", (list, tuple))
        
        self.assert_isinstance(value, "id", six.string_types, is_array=True)
        self._property_id = value

    @schema_property('page')
    def page(self):
        return self._property_page

    @page.setter
    def page(self, value):
        if value is None:
            self._property_page = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "page", six.integer_types)
        self._property_page = value

    @schema_property('page_size')
    def page_size(self):
        return self._property_page_size

    @page_size.setter
    def page_size(self, value):
        if value is None:
            self._property_page_size = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "page_size", six.integer_types)
        self._property_page_size = value

    @schema_property('order_by')
    def order_by(self):
        return self._property_order_by

    @order_by.setter
    def order_by(self, value):
        if value is None:
            self._property_order_by = None
            return
        
        self.assert_isinstance(value, "order_by", (list, tuple))
        
        self.assert_isinstance(value, "order_by", six.string_types, is_array=True)
        self._property_order_by = value

    @schema_property('only_fields')
    def only_fields(self):
        return self._property_only_fields

    @only_fields.setter
    def only_fields(self, value):
        if value is None:
            self._property_only_fields = None
            return
        
        self.assert_isinstance(value, "only_fields", (list, tuple))
        
        self.assert_isinstance(value, "only_fields", six.string_types, is_array=True)
        self._property_only_fields = value


class GetAllResponse(Response):
    """
    Response of storage.get_all endpoint.

    :param results: Storage entries list
    :type results: Sequence[Storage]
    """
    _service = "storage"
    _action = "get_all"
    _version = "1.5"

    _schema = {
        'definitions': {
            'credentials': {
                'properties': {
                    'access_key': {
                        'description': 'Credentials access key',
                        'type': ['string', 'null'],
                    },
                    'secret_key': {
                        'description': 'Credentials secret key',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'storage': {
                'properties': {
                    'company': {
                        'description': 'Company ID',
                        'type': ['string', 'null'],
                    },
                    'created': {
                        'description': 'Entry creation time',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'credentials': {
                        'description': 'Credentials required for accessing the storage',
                        'oneOf': [
                            {'$ref': '#/definitions/credentials'},
                            {'type': 'null'},
                        ],
                    },
                    'id': {'description': 'Entry ID', 'type': ['string', 'null']},
                    'name': {
                        'description': 'Entry name',
                        'type': ['string', 'null'],
                    },
                    'uri': {
                        'description': 'Storage URI',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
        },
        'properties': {
            'results': {
                'description': 'Storage entries list',
                'items': {'$ref': '#/definitions/storage'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, results=None, **kwargs):
        super(GetAllResponse, self).__init__(**kwargs)
        self.results = results

    @schema_property('results')
    def results(self):
        return self._property_results

    @results.setter
    def results(self, value):
        if value is None:
            self._property_results = None
            return
        
        self.assert_isinstance(value, "results", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Storage.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "results", Storage, is_array=True)
        self._property_results = value


response_mapping = {
    GetAllRequest: GetAllResponse,
    CreateRequest: CreateResponse,
    DeleteRequest: DeleteResponse,
}
