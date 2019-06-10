"""
async service

This service provides support for asynchronous API calls.
"""
import six
import types
from datetime import datetime
import enum

from dateutil.parser import parse as parse_datetime

from ....backend_api.session import Request, BatchRequest, Response, DataModel, NonStrictDataModel, CompoundRequest, schema_property, StringEnum


class Call(NonStrictDataModel):
    """
    :param id: The job ID associated with this call.
    :type id: str
    :param status: The job's status.
    :type status: str
    :param created: Job creation time.
    :type created: str
    :param ended: Job end time.
    :type ended: str
    :param enqueued: Job enqueue time.
    :type enqueued: str
    :param meta: Metadata for this job, includes endpoint and additional relevant
        call data.
    :type meta: dict
    :param company: The Company this job belongs to.
    :type company: str
    :param exec_info: Job execution information.
    :type exec_info: str
    """
    _schema = {
        'properties': {
            'company': {
                'description': 'The Company this job belongs to.',
                'type': ['string', 'null'],
            },
            'created': {
                'description': 'Job creation time.',
                'type': ['string', 'null'],
            },
            'ended': {'description': 'Job end time.', 'type': ['string', 'null']},
            'enqueued': {
                'description': 'Job enqueue time.',
                'type': ['string', 'null'],
            },
            'exec_info': {
                'description': 'Job execution information.',
                'type': ['string', 'null'],
            },
            'id': {
                'description': 'The job ID associated with this call.',
                'type': ['string', 'null'],
            },
            'meta': {
                'additionalProperties': True,
                'description': 'Metadata for this job, includes endpoint and additional relevant call data.',
                'type': ['object', 'null'],
            },
            'status': {
                'description': "The job's status.",
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, id=None, status=None, created=None, ended=None, enqueued=None, meta=None, company=None, exec_info=None, **kwargs):
        super(Call, self).__init__(**kwargs)
        self.id = id
        self.status = status
        self.created = created
        self.ended = ended
        self.enqueued = enqueued
        self.meta = meta
        self.company = company
        self.exec_info = exec_info

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

    @schema_property('status')
    def status(self):
        return self._property_status

    @status.setter
    def status(self, value):
        if value is None:
            self._property_status = None
            return
        
        self.assert_isinstance(value, "status", six.string_types)
        self._property_status = value

    @schema_property('created')
    def created(self):
        return self._property_created

    @created.setter
    def created(self, value):
        if value is None:
            self._property_created = None
            return
        
        self.assert_isinstance(value, "created", six.string_types)
        self._property_created = value

    @schema_property('ended')
    def ended(self):
        return self._property_ended

    @ended.setter
    def ended(self, value):
        if value is None:
            self._property_ended = None
            return
        
        self.assert_isinstance(value, "ended", six.string_types)
        self._property_ended = value

    @schema_property('enqueued')
    def enqueued(self):
        return self._property_enqueued

    @enqueued.setter
    def enqueued(self, value):
        if value is None:
            self._property_enqueued = None
            return
        
        self.assert_isinstance(value, "enqueued", six.string_types)
        self._property_enqueued = value

    @schema_property('meta')
    def meta(self):
        return self._property_meta

    @meta.setter
    def meta(self, value):
        if value is None:
            self._property_meta = None
            return
        
        self.assert_isinstance(value, "meta", (dict,))
        self._property_meta = value

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

    @schema_property('exec_info')
    def exec_info(self):
        return self._property_exec_info

    @exec_info.setter
    def exec_info(self, value):
        if value is None:
            self._property_exec_info = None
            return
        
        self.assert_isinstance(value, "exec_info", six.string_types)
        self._property_exec_info = value


class CallsRequest(Request):
    """
    Get a list of all asynchronous API calls handled by the system.
                This includes both previously handled calls, calls being executed and calls waiting in queue.

    :param status: Return only calls who's status is in this list.
    :type status: Sequence[str]
    :param endpoint: Return only calls handling this endpoint. Supports wildcards.
    :type endpoint: str
    :param task: Return only calls associated with this task ID. Supports
        wildcards.
    :type task: str
    """

    _service = "async"
    _action = "calls"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'endpoint': {
                'description': 'Return only calls handling this endpoint. Supports wildcards.',
                'type': ['string', 'null'],
            },
            'status': {
                'description': "Return only calls who's status is in this list.",
                'items': {'enum': ['queued', 'in_progress', 'completed'], 'type': 'string'},
                'type': ['array', 'null'],
            },
            'task': {
                'description': 'Return only calls associated with this task ID. Supports wildcards.',
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, status=None, endpoint=None, task=None, **kwargs):
        super(CallsRequest, self).__init__(**kwargs)
        self.status = status
        self.endpoint = endpoint
        self.task = task

    @schema_property('status')
    def status(self):
        return self._property_status

    @status.setter
    def status(self, value):
        if value is None:
            self._property_status = None
            return
        
        self.assert_isinstance(value, "status", (list, tuple))
        
        self.assert_isinstance(value, "status", six.string_types, is_array=True)
        self._property_status = value

    @schema_property('endpoint')
    def endpoint(self):
        return self._property_endpoint

    @endpoint.setter
    def endpoint(self, value):
        if value is None:
            self._property_endpoint = None
            return
        
        self.assert_isinstance(value, "endpoint", six.string_types)
        self._property_endpoint = value

    @schema_property('task')
    def task(self):
        return self._property_task

    @task.setter
    def task(self, value):
        if value is None:
            self._property_task = None
            return
        
        self.assert_isinstance(value, "task", six.string_types)
        self._property_task = value


class CallsResponse(Response):
    """
    Response of async.calls endpoint.

    :param calls: A list of the current asynchronous calls handled by the system.
    :type calls: Sequence[Call]
    """
    _service = "async"
    _action = "calls"
    _version = "1.5"

    _schema = {
        'definitions': {
            'call': {
                'properties': {
                    'company': {
                        'description': 'The Company this job belongs to.',
                        'type': ['string', 'null'],
                    },
                    'created': {
                        'description': 'Job creation time.',
                        'type': ['string', 'null'],
                    },
                    'ended': {
                        'description': 'Job end time.',
                        'type': ['string', 'null'],
                    },
                    'enqueued': {
                        'description': 'Job enqueue time.',
                        'type': ['string', 'null'],
                    },
                    'exec_info': {
                        'description': 'Job execution information.',
                        'type': ['string', 'null'],
                    },
                    'id': {
                        'description': 'The job ID associated with this call.',
                        'type': ['string', 'null'],
                    },
                    'meta': {
                        'additionalProperties': True,
                        'description': 'Metadata for this job, includes endpoint and additional relevant call data.',
                        'type': ['object', 'null'],
                    },
                    'status': {
                        'description': "The job's status.",
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
        },
        'properties': {
            'calls': {
                'description': 'A list of the current asynchronous calls handled by the system.',
                'items': {'$ref': '#/definitions/call'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, calls=None, **kwargs):
        super(CallsResponse, self).__init__(**kwargs)
        self.calls = calls

    @schema_property('calls')
    def calls(self):
        return self._property_calls

    @calls.setter
    def calls(self, value):
        if value is None:
            self._property_calls = None
            return
        
        self.assert_isinstance(value, "calls", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Call.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "calls", Call, is_array=True)
        self._property_calls = value


class ResultRequest(Request):
    """
    Try getting the result of a previously accepted asynchronous API call.
                If execution for the asynchronous call has completed, the complete call response data will be returned.
                Otherwise, a 202 code will be returned with no data

    :param id: The id returned by the accepted asynchronous API call.
    :type id: str
    """

    _service = "async"
    _action = "result"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'id': {
                'description': 'The id returned by the accepted asynchronous API call.',
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, id=None, **kwargs):
        super(ResultRequest, self).__init__(**kwargs)
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


class ResultResponse(Response):
    """
    Response of async.result endpoint.

    """
    _service = "async"
    _action = "result"
    _version = "1.5"

    _schema = {'additionalProperties': True, 'definitions': {}, 'type': 'object'}


response_mapping = {
    ResultRequest: ResultResponse,
    CallsRequest: CallsResponse,
}
