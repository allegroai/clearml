"""
queues service

Provides a management API for queues of tasks waiting to be executed by workers deployed anywhere (see Workers Service).
"""
import six
from datetime import datetime

from dateutil.parser import parse as parse_datetime

from clearml.backend_api.session import (
    Request,
    Response,
    NonStrictDataModel,
    schema_property,
)


class QueueMetrics(NonStrictDataModel):
    """
    :param queue: ID of the queue
    :type queue: str
    :param dates: List of timestamps (in seconds from epoch) in the acceding order.
        The timestamps are separated by the requested interval. Timestamps where no
        queue status change was recorded are omitted.
    :type dates: Sequence[int]
    :param avg_waiting_times: List of average waiting times for tasks in the queue.
        The points correspond to the timestamps in the dates list. If more than one
        value exists for the given interval then the maximum value is taken.
    :type avg_waiting_times: Sequence[float]
    :param queue_lengths: List of tasks counts in the queue. The points correspond
        to the timestamps in the dates list. If more than one value exists for the
        given interval then the count that corresponds to the maximum average value is
        taken.
    :type queue_lengths: Sequence[int]
    """

    _schema = {
        "properties": {
            "avg_waiting_times": {
                "description": (
                    "List of average waiting times for tasks in the queue. The points correspond to the timestamps in"
                    " the dates list. If more than one value exists for the given interval then the maximum value is"
                    " taken."
                ),
                "items": {"type": "number"},
                "type": ["array", "null"],
            },
            "dates": {
                "description": (
                    "List of timestamps (in seconds from epoch) in the acceding order. The timestamps are separated by"
                    " the requested interval. Timestamps where no queue status change was recorded are omitted."
                ),
                "items": {"type": "integer"},
                "type": ["array", "null"],
            },
            "queue": {"description": "ID of the queue", "type": ["string", "null"]},
            "queue_lengths": {
                "description": (
                    "List of tasks counts in the queue. The points correspond to the timestamps in the dates list. If"
                    " more than one value exists for the given interval then the count that corresponds to the maximum"
                    " average value is taken."
                ),
                "items": {"type": "integer"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, queue=None, dates=None, avg_waiting_times=None, queue_lengths=None, **kwargs):
        super(QueueMetrics, self).__init__(**kwargs)
        self.queue = queue
        self.dates = dates
        self.avg_waiting_times = avg_waiting_times
        self.queue_lengths = queue_lengths

    @schema_property("queue")
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return

        self.assert_isinstance(value, "queue", six.string_types)
        self._property_queue = value

    @schema_property("dates")
    def dates(self):
        return self._property_dates

    @dates.setter
    def dates(self, value):
        if value is None:
            self._property_dates = None
            return

        self.assert_isinstance(value, "dates", (list, tuple))
        value = [int(v) if isinstance(v, float) and v.is_integer() else v for v in value]

        self.assert_isinstance(value, "dates", six.integer_types, is_array=True)
        self._property_dates = value

    @schema_property("avg_waiting_times")
    def avg_waiting_times(self):
        return self._property_avg_waiting_times

    @avg_waiting_times.setter
    def avg_waiting_times(self, value):
        if value is None:
            self._property_avg_waiting_times = None
            return

        self.assert_isinstance(value, "avg_waiting_times", (list, tuple))

        self.assert_isinstance(value, "avg_waiting_times", six.integer_types + (float,), is_array=True)
        self._property_avg_waiting_times = value

    @schema_property("queue_lengths")
    def queue_lengths(self):
        return self._property_queue_lengths

    @queue_lengths.setter
    def queue_lengths(self, value):
        if value is None:
            self._property_queue_lengths = None
            return

        self.assert_isinstance(value, "queue_lengths", (list, tuple))
        value = [int(v) if isinstance(v, float) and v.is_integer() else v for v in value]

        self.assert_isinstance(value, "queue_lengths", six.integer_types, is_array=True)
        self._property_queue_lengths = value


class Entry(NonStrictDataModel):
    """
    :param task: Queued task ID
    :type task: str
    :param added: Time this entry was added to the queue
    :type added: datetime.datetime
    """

    _schema = {
        "properties": {
            "added": {
                "description": "Time this entry was added to the queue",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "task": {"description": "Queued task ID", "type": ["string", "null"]},
        },
        "type": "object",
    }

    def __init__(self, task=None, added=None, **kwargs):
        super(Entry, self).__init__(**kwargs)
        self.task = task
        self.added = added

    @schema_property("task")
    def task(self):
        return self._property_task

    @task.setter
    def task(self, value):
        if value is None:
            self._property_task = None
            return

        self.assert_isinstance(value, "task", six.string_types)
        self._property_task = value

    @schema_property("added")
    def added(self):
        return self._property_added

    @added.setter
    def added(self, value):
        if value is None:
            self._property_added = None
            return

        self.assert_isinstance(value, "added", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_added = value


class MetadataItem(NonStrictDataModel):
    """
    :param key: The key uniquely identifying the metadata item inside the given
        entity
    :type key: str
    :param type: The type of the metadata item
    :type type: str
    :param value: The value stored in the metadata item
    :type value: str
    """

    _schema = {
        "properties": {
            "key": {
                "description": "The key uniquely identifying the metadata item inside the given entity",
                "type": ["string", "null"],
            },
            "type": {
                "description": "The type of the metadata item",
                "type": ["string", "null"],
            },
            "value": {
                "description": "The value stored in the metadata item",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, key=None, type=None, value=None, **kwargs):
        super(MetadataItem, self).__init__(**kwargs)
        self.key = key
        self.type = type
        self.value = value

    @schema_property("key")
    def key(self):
        return self._property_key

    @key.setter
    def key(self, value):
        if value is None:
            self._property_key = None
            return

        self.assert_isinstance(value, "key", six.string_types)
        self._property_key = value

    @schema_property("type")
    def type(self):
        return self._property_type

    @type.setter
    def type(self, value):
        if value is None:
            self._property_type = None
            return

        self.assert_isinstance(value, "type", six.string_types)
        self._property_type = value

    @schema_property("value")
    def value(self):
        return self._property_value

    @value.setter
    def value(self, value):
        if value is None:
            self._property_value = None
            return

        self.assert_isinstance(value, "value", six.string_types)
        self._property_value = value


class Queue(NonStrictDataModel):
    """
    :param id: Queue id
    :type id: str
    :param name: Queue name
    :type name: str
    :param user: Associated user id
    :type user: str
    :param company: Company id
    :type company: str
    :param created: Queue creation time
    :type created: datetime.datetime
    :param tags: User-defined tags
    :type tags: Sequence[str]
    :param system_tags: System tags. This field is reserved for system use, please
        don't use it.
    :type system_tags: Sequence[str]
    :param entries: List of ordered queue entries
    :type entries: Sequence[Entry]
    :param metadata: Queue metadata
    :type metadata: dict
    """

    _schema = {
        "definitions": {
            "metadata_item": {
                "properties": {
                    "key": {
                        "description": "The key uniquely identifying the metadata item inside the given entity",
                        "type": "string",
                    },
                    "type": {
                        "description": "The type of the metadata item",
                        "type": "string",
                    },
                    "value": {
                        "description": "The value stored in the metadata item",
                        "type": "string",
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "company": {"description": "Company id", "type": ["string", "null"]},
            "created": {
                "description": "Queue creation time",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "entries": {
                "description": "List of ordered queue entries",
                "items": {"$ref": "#/definitions/entry"},
                "type": ["array", "null"],
            },
            "id": {"description": "Queue id", "type": ["string", "null"]},
            "metadata": {
                "type": ["object", "null"],
                "items": {"$ref": "#/definitions/metadata_item"},
                "description": "Queue metadata",
            },
            "name": {"description": "Queue name", "type": ["string", "null"]},
            "system_tags": {
                "description": "System tags. This field is reserved for system use, please don't use it.",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "tags": {
                "description": "User-defined tags",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "user": {"description": "Associated user id", "type": ["string", "null"]},
        },
        "type": "object",
    }

    def __init__(
        self,
        id=None,
        name=None,
        user=None,
        company=None,
        created=None,
        tags=None,
        system_tags=None,
        entries=None,
        metadata=None,
        **kwargs
    ):
        super(Queue, self).__init__(**kwargs)
        self.id = id
        self.name = name
        self.user = user
        self.company = company
        self.created = created
        self.tags = tags
        self.system_tags = system_tags
        self.entries = entries
        self.metadata = metadata

    @schema_property("id")
    def id(self):
        return self._property_id

    @id.setter
    def id(self, value):
        if value is None:
            self._property_id = None
            return

        self.assert_isinstance(value, "id", six.string_types)
        self._property_id = value

    @schema_property("name")
    def name(self):
        return self._property_name

    @name.setter
    def name(self, value):
        if value is None:
            self._property_name = None
            return

        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("user")
    def user(self):
        return self._property_user

    @user.setter
    def user(self, value):
        if value is None:
            self._property_user = None
            return

        self.assert_isinstance(value, "user", six.string_types)
        self._property_user = value

    @schema_property("company")
    def company(self):
        return self._property_company

    @company.setter
    def company(self, value):
        if value is None:
            self._property_company = None
            return

        self.assert_isinstance(value, "company", six.string_types)
        self._property_company = value

    @schema_property("created")
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

    @schema_property("tags")
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

    @schema_property("system_tags")
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

    @schema_property("entries")
    def entries(self):
        return self._property_entries

    @entries.setter
    def entries(self, value):
        if value is None:
            self._property_entries = None
            return

        self.assert_isinstance(value, "entries", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Entry.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "entries", Entry, is_array=True)
        self._property_entries = value

    @schema_property("metadata")
    def metadata(self):
        return self._property_metadata

    @metadata.setter
    def metadata(self, value):
        if value is None:
            self._property_metadata = None
            return

        self.assert_isinstance(value, "metadata", (dict,))
        self._property_metadata = value


class AddOrUpdateMetadataRequest(Request):
    """
    Add or update queue metadata

    :param queue: ID of the queue
    :type queue: str
    :param metadata: Metadata items to add or update
    :type metadata: Sequence[MetadataItem]
    :param replace_metadata: If set then the all the metadata items will be
        replaced with the provided ones. Otherwise only the provided metadata items
        will be updated or added
    :type replace_metadata: bool
    """

    _service = "queues"
    _action = "add_or_update_metadata"
    _version = "2.23"
    _schema = {
        "definitions": {
            "metadata_item": {
                "properties": {
                    "key": {
                        "description": "The key uniquely identifying the metadata item inside the given entity",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "The type of the metadata item",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "The value stored in the metadata item",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "metadata": {
                "description": "Metadata items to add or update",
                "items": {"$ref": "#/definitions/metadata_item"},
                "type": "array",
            },
            "queue": {"description": "ID of the queue", "type": "string"},
            "replace_metadata": {
                "default": False,
                "description": (
                    "If set then the all the metadata items will be replaced with the provided ones. Otherwise only the"
                    " provided metadata items will be updated or added"
                ),
                "type": "boolean",
            },
        },
        "required": ["queue", "metadata"],
        "type": "object",
    }

    def __init__(self, queue, metadata, replace_metadata=False, **kwargs):
        super(AddOrUpdateMetadataRequest, self).__init__(**kwargs)
        self.queue = queue
        self.metadata = metadata
        self.replace_metadata = replace_metadata

    @schema_property("queue")
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return

        self.assert_isinstance(value, "queue", six.string_types)
        self._property_queue = value

    @schema_property("metadata")
    def metadata(self):
        return self._property_metadata

    @metadata.setter
    def metadata(self, value):
        if value is None:
            self._property_metadata = None
            return

        self.assert_isinstance(value, "metadata", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [MetadataItem.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "metadata", MetadataItem, is_array=True)
        self._property_metadata = value

    @schema_property("replace_metadata")
    def replace_metadata(self):
        return self._property_replace_metadata

    @replace_metadata.setter
    def replace_metadata(self, value):
        if value is None:
            self._property_replace_metadata = None
            return

        self.assert_isinstance(value, "replace_metadata", (bool,))
        self._property_replace_metadata = value


class AddOrUpdateMetadataResponse(Response):
    """
    Response of queues.add_or_update_metadata endpoint.

    :param updated: Number of queues updated (0 or 1)
    :type updated: int
    """

    _service = "queues"
    _action = "add_or_update_metadata"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "updated": {
                "description": "Number of queues updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, updated=None, **kwargs):
        super(AddOrUpdateMetadataResponse, self).__init__(**kwargs)
        self.updated = updated

    @schema_property("updated")
    def updated(self):
        return self._property_updated

    @updated.setter
    def updated(self, value):
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value


class AddTaskRequest(Request):
    """
    Adds a task entry to the queue.

    :param queue: Queue id
    :type queue: str
    :param task: Task id
    :type task: str
    """

    _service = "queues"
    _action = "add_task"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "queue": {"description": "Queue id", "type": "string"},
            "task": {"description": "Task id", "type": "string"},
        },
        "required": ["queue", "task"],
        "type": "object",
    }

    def __init__(self, queue, task, **kwargs):
        super(AddTaskRequest, self).__init__(**kwargs)
        self.queue = queue
        self.task = task

    @schema_property("queue")
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return

        self.assert_isinstance(value, "queue", six.string_types)
        self._property_queue = value

    @schema_property("task")
    def task(self):
        return self._property_task

    @task.setter
    def task(self, value):
        if value is None:
            self._property_task = None
            return

        self.assert_isinstance(value, "task", six.string_types)
        self._property_task = value


class AddTaskResponse(Response):
    """
    Response of queues.add_task endpoint.

    :param added: Number of tasks added (0 or 1)
    :type added: int
    """

    _service = "queues"
    _action = "add_task"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "added": {
                "description": "Number of tasks added (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, added=None, **kwargs):
        super(AddTaskResponse, self).__init__(**kwargs)
        self.added = added

    @schema_property("added")
    def added(self):
        return self._property_added

    @added.setter
    def added(self, value):
        if value is None:
            self._property_added = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "added", six.integer_types)
        self._property_added = value


class CreateRequest(Request):
    """
    Create a new queue

    :param name: Queue name Unique within the company.
    :type name: str
    :param tags: User-defined tags list
    :type tags: Sequence[str]
    :param system_tags: System tags list. This field is reserved for system use,
        please don't use it.
    :type system_tags: Sequence[str]
    """

    _service = "queues"
    _action = "create"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "name": {
                "description": "Queue name Unique within the company.",
                "type": "string",
            },
            "system_tags": {
                "description": "System tags list. This field is reserved for system use, please don't use it.",
                "items": {"type": "string"},
                "type": "array",
            },
            "tags": {
                "description": "User-defined tags list",
                "items": {"type": "string"},
                "type": "array",
            },
        },
        "required": ["name"],
        "type": "object",
    }

    def __init__(self, name, tags=None, system_tags=None, **kwargs):
        super(CreateRequest, self).__init__(**kwargs)
        self.name = name
        self.tags = tags
        self.system_tags = system_tags

    @schema_property("name")
    def name(self):
        return self._property_name

    @name.setter
    def name(self, value):
        if value is None:
            self._property_name = None
            return

        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("tags")
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

    @schema_property("system_tags")
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


class CreateResponse(Response):
    """
    Response of queues.create endpoint.

    :param id: New queue ID
    :type id: str
    """

    _service = "queues"
    _action = "create"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {"id": {"description": "New queue ID", "type": ["string", "null"]}},
        "type": "object",
    }

    def __init__(self, id=None, **kwargs):
        super(CreateResponse, self).__init__(**kwargs)
        self.id = id

    @schema_property("id")
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
    Deletes a queue. If the queue is not empty and force is not set to true, queue will not be deleted.

    :param queue: Queue id
    :type queue: str
    :param force: Force delete of non-empty queue. Defaults to false
    :type force: bool
    """

    _service = "queues"
    _action = "delete"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "force": {
                "default": False,
                "description": "Force delete of non-empty queue. Defaults to false",
                "type": "boolean",
            },
            "queue": {"description": "Queue id", "type": "string"},
        },
        "required": ["queue"],
        "type": "object",
    }

    def __init__(self, queue, force=False, **kwargs):
        super(DeleteRequest, self).__init__(**kwargs)
        self.queue = queue
        self.force = force

    @schema_property("queue")
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return

        self.assert_isinstance(value, "queue", six.string_types)
        self._property_queue = value

    @schema_property("force")
    def force(self):
        return self._property_force

    @force.setter
    def force(self, value):
        if value is None:
            self._property_force = None
            return

        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value


class DeleteResponse(Response):
    """
    Response of queues.delete endpoint.

    :param deleted: Number of queues deleted (0 or 1)
    :type deleted: int
    """

    _service = "queues"
    _action = "delete"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "deleted": {
                "description": "Number of queues deleted (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, deleted=None, **kwargs):
        super(DeleteResponse, self).__init__(**kwargs)
        self.deleted = deleted

    @schema_property("deleted")
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


class DeleteMetadataRequest(Request):
    """
    Delete metadata from queue

    :param queue: ID of the queue
    :type queue: str
    :param keys: The list of metadata keys to delete
    :type keys: Sequence[str]
    """

    _service = "queues"
    _action = "delete_metadata"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "keys": {
                "description": "The list of metadata keys to delete",
                "items": {"type": "string"},
                "type": "array",
            },
            "queue": {"description": "ID of the queue", "type": "string"},
        },
        "required": ["queue", "keys"],
        "type": "object",
    }

    def __init__(self, queue, keys, **kwargs):
        super(DeleteMetadataRequest, self).__init__(**kwargs)
        self.queue = queue
        self.keys = keys

    @schema_property("queue")
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return

        self.assert_isinstance(value, "queue", six.string_types)
        self._property_queue = value

    @schema_property("keys")
    def keys(self):
        return self._property_keys

    @keys.setter
    def keys(self, value):
        if value is None:
            self._property_keys = None
            return

        self.assert_isinstance(value, "keys", (list, tuple))

        self.assert_isinstance(value, "keys", six.string_types, is_array=True)
        self._property_keys = value


class DeleteMetadataResponse(Response):
    """
    Response of queues.delete_metadata endpoint.

    :param updated: Number of queues updated (0 or 1)
    :type updated: int
    """

    _service = "queues"
    _action = "delete_metadata"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "updated": {
                "description": "Number of queues updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, updated=None, **kwargs):
        super(DeleteMetadataResponse, self).__init__(**kwargs)
        self.updated = updated

    @schema_property("updated")
    def updated(self):
        return self._property_updated

    @updated.setter
    def updated(self, value):
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value


class GetAllRequest(Request):
    """
    Get all queues

    :param name: Get only queues whose name matches this pattern (python regular
        expression syntax)
    :type name: str
    :param id: List of Queue IDs used to filter results
    :type id: Sequence[str]
    :param tags: User-defined tags list used to filter results. Prepend '-' to tag
        name to indicate exclusion
    :type tags: Sequence[str]
    :param system_tags: System tags list used to filter results. Prepend '-' to
        system tag name to indicate exclusion
    :type system_tags: Sequence[str]
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
    :param search_text: Free text search query
    :type search_text: str
    :param only_fields: List of document field names (nesting is supported using
        '.', e.g. execution.model_labels). If provided, this list defines the query's
        projection (only these fields will be returned for each result entry)
    :type only_fields: Sequence[str]
    :param scroll_id: Scroll ID returned from the previos calls to get_all
    :type scroll_id: str
    :param refresh_scroll: If set then all the data received with this scroll will
        be requeried
    :type refresh_scroll: bool
    :param size: The number of queues to retrieve
    :type size: int
    :param max_task_entries: Max number of queue task entries to return
    :type max_task_entries: int
    :param search_hidden: If set to 'true' then hidden queues are included in the
        search results
    :type search_hidden: bool
    """

    _service = "queues"
    _action = "get_all"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "id": {
                "description": "List of Queue IDs used to filter results",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "max_task_entries": {
                "description": "Max number of queue task entries to return",
                "type": ["integer", "null"],
            },
            "name": {
                "description": "Get only queues whose name matches this pattern (python regular expression syntax)",
                "type": ["string", "null"],
            },
            "only_fields": {
                "description": (
                    "List of document field names (nesting is supported using '.', e.g. execution.model_labels). If"
                    " provided, this list defines the query's projection (only these fields will be returned for each"
                    " result entry)"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "order_by": {
                "description": (
                    "List of field names to order by. When search_text is used, '@text_score' can be used as a field"
                    " representing the text score of returned documents. Use '-' prefix to specify descending order."
                    " Optional, recommended when using page"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "page": {
                "description": "Page number, returns a specific page out of the result list of results.",
                "minimum": 0,
                "type": ["integer", "null"],
            },
            "page_size": {
                "description": (
                    "Page size, specifies the number of results returned in each page (last page may contain fewer "
                    "results)"
                ),
                "minimum": 1,
                "type": ["integer", "null"],
            },
            "refresh_scroll": {
                "description": "If set then all the data received with this scroll will be requeried",
                "type": ["boolean", "null"],
            },
            "scroll_id": {
                "description": "Scroll ID returned from the previos calls to get_all",
                "type": ["string", "null"],
            },
            "search_hidden": {
                "default": False,
                "description": "If set to 'true' then hidden queues are included in the search results",
                "type": ["boolean", "null"],
            },
            "search_text": {
                "description": "Free text search query",
                "type": ["string", "null"],
            },
            "size": {
                "description": "The number of queues to retrieve",
                "minimum": 1,
                "type": ["integer", "null"],
            },
            "system_tags": {
                "description": (
                    "System tags list used to filter results. Prepend '-' to system tag name to indicate exclusion"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "tags": {
                "description": (
                    "User-defined tags list used to filter results. Prepend '-' to tag name to indicate exclusion"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        name=None,
        id=None,
        tags=None,
        system_tags=None,
        page=None,
        page_size=None,
        order_by=None,
        search_text=None,
        only_fields=None,
        scroll_id=None,
        refresh_scroll=None,
        size=None,
        max_task_entries=None,
        search_hidden=False,
        **kwargs
    ):
        super(GetAllRequest, self).__init__(**kwargs)
        self.name = name
        self.id = id
        self.tags = tags
        self.system_tags = system_tags
        self.page = page
        self.page_size = page_size
        self.order_by = order_by
        self.search_text = search_text
        self.only_fields = only_fields
        self.scroll_id = scroll_id
        self.refresh_scroll = refresh_scroll
        self.size = size
        self.max_task_entries = max_task_entries
        self.search_hidden = search_hidden

    @schema_property("name")
    def name(self):
        return self._property_name

    @name.setter
    def name(self, value):
        if value is None:
            self._property_name = None
            return

        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("id")
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

    @schema_property("tags")
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

    @schema_property("system_tags")
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

    @schema_property("page")
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

    @schema_property("page_size")
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

    @schema_property("order_by")
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

    @schema_property("search_text")
    def search_text(self):
        return self._property_search_text

    @search_text.setter
    def search_text(self, value):
        if value is None:
            self._property_search_text = None
            return

        self.assert_isinstance(value, "search_text", six.string_types)
        self._property_search_text = value

    @schema_property("only_fields")
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

    @schema_property("scroll_id")
    def scroll_id(self):
        return self._property_scroll_id

    @scroll_id.setter
    def scroll_id(self, value):
        if value is None:
            self._property_scroll_id = None
            return

        self.assert_isinstance(value, "scroll_id", six.string_types)
        self._property_scroll_id = value

    @schema_property("refresh_scroll")
    def refresh_scroll(self):
        return self._property_refresh_scroll

    @refresh_scroll.setter
    def refresh_scroll(self, value):
        if value is None:
            self._property_refresh_scroll = None
            return

        self.assert_isinstance(value, "refresh_scroll", (bool,))
        self._property_refresh_scroll = value

    @schema_property("size")
    def size(self):
        return self._property_size

    @size.setter
    def size(self, value):
        if value is None:
            self._property_size = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "size", six.integer_types)
        self._property_size = value

    @schema_property("max_task_entries")
    def max_task_entries(self):
        return self._property_max_task_entries

    @max_task_entries.setter
    def max_task_entries(self, value):
        if value is None:
            self._property_max_task_entries = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "max_task_entries", six.integer_types)
        self._property_max_task_entries = value

    @schema_property("search_hidden")
    def search_hidden(self):
        return self._property_search_hidden

    @search_hidden.setter
    def search_hidden(self, value):
        if value is None:
            self._property_search_hidden = None
            return

        self.assert_isinstance(value, "search_hidden", (bool,))
        self._property_search_hidden = value


class GetAllResponse(Response):
    """
    Response of queues.get_all endpoint.

    :param queues: Queues list
    :type queues: Sequence[Queue]
    :param scroll_id: Scroll ID that can be used with the next calls to get_all to
        retrieve more data
    :type scroll_id: str
    """

    _service = "queues"
    _action = "get_all"
    _version = "2.23"

    _schema = {
        "definitions": {
            "entry": {
                "properties": {
                    "added": {
                        "description": "Time this entry was added to the queue",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "task": {
                        "description": "Queued task ID",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "metadata_item": {
                "properties": {
                    "key": {
                        "description": "The key uniquely identifying the metadata item inside the given entity",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "The type of the metadata item",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "The value stored in the metadata item",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "queue": {
                "properties": {
                    "company": {
                        "description": "Company id",
                        "type": ["string", "null"],
                    },
                    "created": {
                        "description": "Queue creation time",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "entries": {
                        "description": "List of ordered queue entries",
                        "items": {"$ref": "#/definitions/entry"},
                        "type": ["array", "null"],
                    },
                    "id": {"description": "Queue id", "type": ["string", "null"]},
                    "metadata": {
                        "additionalProperties": {"$ref": "#/definitions/metadata_item"},
                        "description": "Queue metadata",
                        "type": ["object", "null"],
                    },
                    "name": {"description": "Queue name", "type": ["string", "null"]},
                    "system_tags": {
                        "description": "System tags. This field is reserved for system use, please don't use it.",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "tags": {
                        "description": "User-defined tags",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "user": {
                        "description": "Associated user id",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "queues": {
                "description": "Queues list",
                "items": {"$ref": "#/definitions/queue"},
                "type": ["array", "null"],
            },
            "scroll_id": {
                "description": "Scroll ID that can be used with the next calls to get_all to retrieve more data",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, queues=None, scroll_id=None, **kwargs):
        super(GetAllResponse, self).__init__(**kwargs)
        self.queues = queues
        self.scroll_id = scroll_id

    @schema_property("queues")
    def queues(self):
        return self._property_queues

    @queues.setter
    def queues(self, value):
        if value is None:
            self._property_queues = None
            return

        self.assert_isinstance(value, "queues", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Queue.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "queues", Queue, is_array=True)
        self._property_queues = value

    @schema_property("scroll_id")
    def scroll_id(self):
        return self._property_scroll_id

    @scroll_id.setter
    def scroll_id(self, value):
        if value is None:
            self._property_scroll_id = None
            return

        self.assert_isinstance(value, "scroll_id", six.string_types)
        self._property_scroll_id = value


class GetByIdRequest(Request):
    """
    Gets queue information

    :param queue: Queue ID
    :type queue: str
    :param max_task_entries: Max number of queue task entries to return
    :type max_task_entries: int
    """

    _service = "queues"
    _action = "get_by_id"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "max_task_entries": {
                "description": "Max number of queue task entries to return",
                "type": "integer",
            },
            "queue": {"description": "Queue ID", "type": "string"},
        },
        "required": ["queue"],
        "type": "object",
    }

    def __init__(self, queue, max_task_entries=None, **kwargs):
        super(GetByIdRequest, self).__init__(**kwargs)
        self.queue = queue
        self.max_task_entries = max_task_entries

    @schema_property("queue")
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return

        self.assert_isinstance(value, "queue", six.string_types)
        self._property_queue = value

    @schema_property("max_task_entries")
    def max_task_entries(self):
        return self._property_max_task_entries

    @max_task_entries.setter
    def max_task_entries(self, value):
        if value is None:
            self._property_max_task_entries = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "max_task_entries", six.integer_types)
        self._property_max_task_entries = value


class GetByIdResponse(Response):
    """
    Response of queues.get_by_id endpoint.

    :param queue: Queue info
    :type queue: Queue
    """

    _service = "queues"
    _action = "get_by_id"
    _version = "2.23"

    _schema = {
        "definitions": {
            "entry": {
                "properties": {
                    "added": {
                        "description": "Time this entry was added to the queue",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "task": {
                        "description": "Queued task ID",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "metadata_item": {
                "properties": {
                    "key": {
                        "description": "The key uniquely identifying the metadata item inside the given entity",
                        "type": ["string", "null"],
                    },
                    "type": {
                        "description": "The type of the metadata item",
                        "type": ["string", "null"],
                    },
                    "value": {
                        "description": "The value stored in the metadata item",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
            "queue": {
                "properties": {
                    "company": {
                        "description": "Company id",
                        "type": ["string", "null"],
                    },
                    "created": {
                        "description": "Queue creation time",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "entries": {
                        "description": "List of ordered queue entries",
                        "items": {"$ref": "#/definitions/entry"},
                        "type": ["array", "null"],
                    },
                    "id": {"description": "Queue id", "type": ["string", "null"]},
                    "metadata": {
                        "additionalProperties": {"$ref": "#/definitions/metadata_item"},
                        "description": "Queue metadata",
                        "type": ["object", "null"],
                    },
                    "name": {"description": "Queue name", "type": ["string", "null"]},
                    "system_tags": {
                        "description": "System tags. This field is reserved for system use, please don't use it.",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "tags": {
                        "description": "User-defined tags",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "user": {
                        "description": "Associated user id",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "queue": {
                "description": "Queue info",
                "oneOf": [{"$ref": "#/definitions/queue"}, {"type": "null"}],
            }
        },
        "type": "object",
    }

    def __init__(self, queue=None, **kwargs):
        super(GetByIdResponse, self).__init__(**kwargs)
        self.queue = queue

    @schema_property("queue")
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return
        if isinstance(value, dict):
            value = Queue.from_dict(value)
        else:
            self.assert_isinstance(value, "queue", Queue)
        self._property_queue = value


class GetDefaultRequest(Request):
    """ """

    _service = "queues"
    _action = "get_default"
    _version = "2.23"
    _schema = {
        "additionalProperties": False,
        "definitions": {},
        "properties": {},
        "type": "object",
    }


class GetDefaultResponse(Response):
    """
    Response of queues.get_default endpoint.

    :param id: Queue id
    :type id: str
    :param name: Queue name
    :type name: str
    """

    _service = "queues"
    _action = "get_default"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "id": {"description": "Queue id", "type": ["string", "null"]},
            "name": {"description": "Queue name", "type": ["string", "null"]},
        },
        "type": "object",
    }

    def __init__(self, id=None, name=None, **kwargs):
        super(GetDefaultResponse, self).__init__(**kwargs)
        self.id = id
        self.name = name

    @schema_property("id")
    def id(self):
        return self._property_id

    @id.setter
    def id(self, value):
        if value is None:
            self._property_id = None
            return

        self.assert_isinstance(value, "id", six.string_types)
        self._property_id = value

    @schema_property("name")
    def name(self):
        return self._property_name

    @name.setter
    def name(self, value):
        if value is None:
            self._property_name = None
            return

        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value


class GetNextTaskRequest(Request):
    """
    Gets the next task from the top of the queue (FIFO). The task entry is removed from the queue.

    :param queue: Queue id
    :type queue: str
    :param get_task_info: If set then additional task info is returned
    :type get_task_info: bool
    :param task: Task company ID
    :type task: str
    """

    _service = "queues"
    _action = "get_next_task"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "get_task_info": {
                "default": False,
                "description": "If set then additional task info is returned",
                "type": "boolean",
            },
            "queue": {"description": "Queue id", "type": "string"},
            "task": {"description": "Task company ID", "type": "string"},
        },
        "required": ["queue"],
        "type": "object",
    }

    def __init__(self, queue, get_task_info=False, task=None, **kwargs):
        super(GetNextTaskRequest, self).__init__(**kwargs)
        self.queue = queue
        self.get_task_info = get_task_info
        self.task = task

    @schema_property("queue")
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return

        self.assert_isinstance(value, "queue", six.string_types)
        self._property_queue = value

    @schema_property("get_task_info")
    def get_task_info(self):
        return self._property_get_task_info

    @get_task_info.setter
    def get_task_info(self, value):
        if value is None:
            self._property_get_task_info = None
            return

        self.assert_isinstance(value, "get_task_info", (bool,))
        self._property_get_task_info = value

    @schema_property("task")
    def task(self):
        return self._property_task

    @task.setter
    def task(self, value):
        if value is None:
            self._property_task = None
            return

        self.assert_isinstance(value, "task", six.string_types)
        self._property_task = value


class GetNextTaskResponse(Response):
    """
    Response of queues.get_next_task endpoint.

    :param entry: Entry information
    :type entry: Entry
    :param task_info: Info about the returned task. Returned only if get_task_info
        is set to True
    :type task_info: dict
    """

    _service = "queues"
    _action = "get_next_task"
    _version = "2.23"

    _schema = {
        "definitions": {
            "entry": {
                "properties": {
                    "added": {
                        "description": "Time this entry was added to the queue",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "task": {
                        "description": "Queued task ID",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "entry": {
                "description": "Entry information",
                "oneOf": [{"$ref": "#/definitions/entry"}, {"type": "null"}],
            },
            "task_info": {
                "description": "Info about the returned task. Returned only if get_task_info is set to True",
                "properties": {
                    "company": {
                        "description": "Task company ID",
                        "type": "string",
                    },
                    "user": {
                        "description": "ID of the user who created the task",
                        "type": "string",
                    },
                },
                "type": ["object", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, entry=None, task_info=None, **kwargs):
        super(GetNextTaskResponse, self).__init__(**kwargs)
        self.entry = entry
        self.task_info = task_info

    @schema_property("entry")
    def entry(self):
        return self._property_entry

    @entry.setter
    def entry(self, value):
        if value is None:
            self._property_entry = None
            return
        if isinstance(value, dict):
            value = Entry.from_dict(value)
        else:
            self.assert_isinstance(value, "entry", Entry)
        self._property_entry = value

    @schema_property("task_info")
    def task_info(self):
        return self._property_task_info

    @task_info.setter
    def task_info(self, value):
        if value is None:
            self._property_task_info = None
            return

        self.assert_isinstance(value, "task_info", (dict,))
        self._property_task_info = value


class GetNumEntriesRequest(Request):
    """
    Get the number of task entries in the given queue

    :param queue: ID of the queue
    :type queue: str
    """

    _service = "queues"
    _action = "get_num_entries"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {"queue": {"description": "ID of the queue", "type": "string"}},
        "required": ["queue"],
        "type": "object",
    }

    def __init__(self, queue, **kwargs):
        super(GetNumEntriesRequest, self).__init__(**kwargs)
        self.queue = queue

    @schema_property("queue")
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return

        self.assert_isinstance(value, "queue", six.string_types)
        self._property_queue = value


class GetNumEntriesResponse(Response):
    """
    Response of queues.get_num_entries endpoint.

    :param num: Number of entries
    :type num: int
    """

    _service = "queues"
    _action = "get_num_entries"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {"num": {"description": "Number of entries", "type": ["integer", "null"]}},
        "type": "object",
    }

    def __init__(self, num=None, **kwargs):
        super(GetNumEntriesResponse, self).__init__(**kwargs)
        self.num = num

    @schema_property("num")
    def num(self):
        return self._property_num

    @num.setter
    def num(self, value):
        if value is None:
            self._property_num = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "num", six.integer_types)
        self._property_num = value


class GetQueueMetricsRequest(Request):
    """
    Returns metrics of the company queues. The metrics are avaraged in the specified interval.

    :param from_date: Starting time (in seconds from epoch) for collecting metrics
    :type from_date: float
    :param to_date: Ending time (in seconds from epoch) for collecting metrics
    :type to_date: float
    :param interval: Time interval in seconds for a single metrics point. The
        minimal value is 1
    :type interval: int
    :param queue_ids: List of queue ids to collect metrics for. If not provided or
        empty then all then average metrics across all the company queues will be
        returned.
    :type queue_ids: Sequence[str]
    :param refresh: If set then the new queue metrics is taken
    :type refresh: bool
    """

    _service = "queues"
    _action = "get_queue_metrics"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "from_date": {
                "description": "Starting time (in seconds from epoch) for collecting metrics",
                "type": "number",
            },
            "interval": {
                "description": "Time interval in seconds for a single metrics point. The minimal value is 1",
                "type": "integer",
            },
            "queue_ids": {
                "description": (
                    "List of queue ids to collect metrics for. If not provided or empty then all then average metrics"
                    " across all the company queues will be returned."
                ),
                "items": {"type": "string"},
                "type": "array",
            },
            "refresh": {
                "default": False,
                "description": "If set then the new queue metrics is taken",
                "type": "boolean",
            },
            "to_date": {
                "description": "Ending time (in seconds from epoch) for collecting metrics",
                "type": "number",
            },
        },
        "required": ["from_date", "to_date", "interval"],
        "type": "object",
    }

    def __init__(self, from_date, to_date, interval, queue_ids=None, refresh=False, **kwargs):
        super(GetQueueMetricsRequest, self).__init__(**kwargs)
        self.from_date = from_date
        self.to_date = to_date
        self.interval = interval
        self.queue_ids = queue_ids
        self.refresh = refresh

    @schema_property("from_date")
    def from_date(self):
        return self._property_from_date

    @from_date.setter
    def from_date(self, value):
        if value is None:
            self._property_from_date = None
            return

        self.assert_isinstance(value, "from_date", six.integer_types + (float,))
        self._property_from_date = value

    @schema_property("to_date")
    def to_date(self):
        return self._property_to_date

    @to_date.setter
    def to_date(self, value):
        if value is None:
            self._property_to_date = None
            return

        self.assert_isinstance(value, "to_date", six.integer_types + (float,))
        self._property_to_date = value

    @schema_property("interval")
    def interval(self):
        return self._property_interval

    @interval.setter
    def interval(self, value):
        if value is None:
            self._property_interval = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "interval", six.integer_types)
        self._property_interval = value

    @schema_property("queue_ids")
    def queue_ids(self):
        return self._property_queue_ids

    @queue_ids.setter
    def queue_ids(self, value):
        if value is None:
            self._property_queue_ids = None
            return

        self.assert_isinstance(value, "queue_ids", (list, tuple))

        self.assert_isinstance(value, "queue_ids", six.string_types, is_array=True)
        self._property_queue_ids = value

    @schema_property("refresh")
    def refresh(self):
        return self._property_refresh

    @refresh.setter
    def refresh(self, value):
        if value is None:
            self._property_refresh = None
            return

        self.assert_isinstance(value, "refresh", (bool,))
        self._property_refresh = value


class GetQueueMetricsResponse(Response):
    """
    Response of queues.get_queue_metrics endpoint.

    :param queues: List of the requested queues with their metrics. If no queue ids
        were requested then 'all' queue is returned with the metrics averaged accross
        all the company queues.
    :type queues: Sequence[QueueMetrics]
    """

    _service = "queues"
    _action = "get_queue_metrics"
    _version = "2.23"

    _schema = {
        "definitions": {
            "queue_metrics": {
                "properties": {
                    "avg_waiting_times": {
                        "description": (
                            "List of average waiting times for tasks in the queue. The points correspond to the"
                            " timestamps in the dates list. If more than one value exists for the given interval then"
                            " the maximum value is taken."
                        ),
                        "items": {"type": "number"},
                        "type": ["array", "null"],
                    },
                    "dates": {
                        "description": (
                            "List of timestamps (in seconds from epoch) in the acceding order. The timestamps are"
                            " separated by the requested interval. Timestamps where no queue status change was recorded"
                            " are omitted."
                        ),
                        "items": {"type": "integer"},
                        "type": ["array", "null"],
                    },
                    "queue": {
                        "description": "ID of the queue",
                        "type": ["string", "null"],
                    },
                    "queue_lengths": {
                        "description": (
                            "List of tasks counts in the queue. The points correspond to the timestamps in the dates"
                            " list. If more than one value exists for the given interval then the count that"
                            " corresponds to the maximum average value is taken."
                        ),
                        "items": {"type": "integer"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "queues": {
                "description": (
                    "List of the requested queues with their metrics. If no queue ids were requested then 'all' queue"
                    " is returned with the metrics averaged accross all the company queues."
                ),
                "items": {"$ref": "#/definitions/queue_metrics"},
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, queues=None, **kwargs):
        super(GetQueueMetricsResponse, self).__init__(**kwargs)
        self.queues = queues

    @schema_property("queues")
    def queues(self):
        return self._property_queues

    @queues.setter
    def queues(self, value):
        if value is None:
            self._property_queues = None
            return

        self.assert_isinstance(value, "queues", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [QueueMetrics.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "queues", QueueMetrics, is_array=True)
        self._property_queues = value


class MoveTaskBackwardRequest(Request):
    """
    :param queue: Queue id
    :type queue: str
    :param task: Task id
    :type task: str
    :param count: Number of positions in the queue to move the task forward
        relative to the current position. Optional, the default value is 1.
    :type count: int
    """

    _service = "queues"
    _action = "move_task_backward"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "count": {
                "description": (
                    "Number of positions in the queue to move the task forward relative to the current position."
                    " Optional, the default value is 1."
                ),
                "type": "integer",
            },
            "queue": {"description": "Queue id", "type": "string"},
            "task": {"description": "Task id", "type": "string"},
        },
        "required": ["queue", "task"],
        "type": "object",
    }

    def __init__(self, queue, task, count=None, **kwargs):
        super(MoveTaskBackwardRequest, self).__init__(**kwargs)
        self.queue = queue
        self.task = task
        self.count = count

    @schema_property("queue")
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return

        self.assert_isinstance(value, "queue", six.string_types)
        self._property_queue = value

    @schema_property("task")
    def task(self):
        return self._property_task

    @task.setter
    def task(self, value):
        if value is None:
            self._property_task = None
            return

        self.assert_isinstance(value, "task", six.string_types)
        self._property_task = value

    @schema_property("count")
    def count(self):
        return self._property_count

    @count.setter
    def count(self, value):
        if value is None:
            self._property_count = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "count", six.integer_types)
        self._property_count = value


class MoveTaskBackwardResponse(Response):
    """
    Response of queues.move_task_backward endpoint.

    :param position: The new position of the task entry in the queue (index, -1
        represents bottom of queue)
    :type position: int
    """

    _service = "queues"
    _action = "move_task_backward"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "position": {
                "description": "The new position of the task entry in the queue (index, -1 represents bottom of queue)",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, position=None, **kwargs):
        super(MoveTaskBackwardResponse, self).__init__(**kwargs)
        self.position = position

    @schema_property("position")
    def position(self):
        return self._property_position

    @position.setter
    def position(self, value):
        if value is None:
            self._property_position = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "position", six.integer_types)
        self._property_position = value


class MoveTaskForwardRequest(Request):
    """
    Moves a task entry one step forward towards the top of the queue.

    :param queue: Queue id
    :type queue: str
    :param task: Task id
    :type task: str
    :param count: Number of positions in the queue to move the task forward
        relative to the current position. Optional, the default value is 1.
    :type count: int
    """

    _service = "queues"
    _action = "move_task_forward"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "count": {
                "description": (
                    "Number of positions in the queue to move the task forward relative to the current position."
                    " Optional, the default value is 1."
                ),
                "type": "integer",
            },
            "queue": {"description": "Queue id", "type": "string"},
            "task": {"description": "Task id", "type": "string"},
        },
        "required": ["queue", "task"],
        "type": "object",
    }

    def __init__(self, queue, task, count=None, **kwargs):
        super(MoveTaskForwardRequest, self).__init__(**kwargs)
        self.queue = queue
        self.task = task
        self.count = count

    @schema_property("queue")
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return

        self.assert_isinstance(value, "queue", six.string_types)
        self._property_queue = value

    @schema_property("task")
    def task(self):
        return self._property_task

    @task.setter
    def task(self, value):
        if value is None:
            self._property_task = None
            return

        self.assert_isinstance(value, "task", six.string_types)
        self._property_task = value

    @schema_property("count")
    def count(self):
        return self._property_count

    @count.setter
    def count(self, value):
        if value is None:
            self._property_count = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "count", six.integer_types)
        self._property_count = value


class MoveTaskForwardResponse(Response):
    """
    Response of queues.move_task_forward endpoint.

    :param position: The new position of the task entry in the queue (index, -1
        represents bottom of queue)
    :type position: int
    """

    _service = "queues"
    _action = "move_task_forward"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "position": {
                "description": "The new position of the task entry in the queue (index, -1 represents bottom of queue)",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, position=None, **kwargs):
        super(MoveTaskForwardResponse, self).__init__(**kwargs)
        self.position = position

    @schema_property("position")
    def position(self):
        return self._property_position

    @position.setter
    def position(self, value):
        if value is None:
            self._property_position = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "position", six.integer_types)
        self._property_position = value


class MoveTaskToBackRequest(Request):
    """
    :param queue: Queue id
    :type queue: str
    :param task: Task id
    :type task: str
    """

    _service = "queues"
    _action = "move_task_to_back"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "queue": {"description": "Queue id", "type": "string"},
            "task": {"description": "Task id", "type": "string"},
        },
        "required": ["queue", "task"],
        "type": "object",
    }

    def __init__(self, queue, task, **kwargs):
        super(MoveTaskToBackRequest, self).__init__(**kwargs)
        self.queue = queue
        self.task = task

    @schema_property("queue")
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return

        self.assert_isinstance(value, "queue", six.string_types)
        self._property_queue = value

    @schema_property("task")
    def task(self):
        return self._property_task

    @task.setter
    def task(self, value):
        if value is None:
            self._property_task = None
            return

        self.assert_isinstance(value, "task", six.string_types)
        self._property_task = value


class MoveTaskToBackResponse(Response):
    """
    Response of queues.move_task_to_back endpoint.

    :param position: The new position of the task entry in the queue (index, -1
        represents bottom of queue)
    :type position: int
    """

    _service = "queues"
    _action = "move_task_to_back"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "position": {
                "description": "The new position of the task entry in the queue (index, -1 represents bottom of queue)",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, position=None, **kwargs):
        super(MoveTaskToBackResponse, self).__init__(**kwargs)
        self.position = position

    @schema_property("position")
    def position(self):
        return self._property_position

    @position.setter
    def position(self, value):
        if value is None:
            self._property_position = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "position", six.integer_types)
        self._property_position = value


class MoveTaskToFrontRequest(Request):
    """
    :param queue: Queue id
    :type queue: str
    :param task: Task id
    :type task: str
    """

    _service = "queues"
    _action = "move_task_to_front"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "queue": {"description": "Queue id", "type": "string"},
            "task": {"description": "Task id", "type": "string"},
        },
        "required": ["queue", "task"],
        "type": "object",
    }

    def __init__(self, queue, task, **kwargs):
        super(MoveTaskToFrontRequest, self).__init__(**kwargs)
        self.queue = queue
        self.task = task

    @schema_property("queue")
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return

        self.assert_isinstance(value, "queue", six.string_types)
        self._property_queue = value

    @schema_property("task")
    def task(self):
        return self._property_task

    @task.setter
    def task(self, value):
        if value is None:
            self._property_task = None
            return

        self.assert_isinstance(value, "task", six.string_types)
        self._property_task = value


class MoveTaskToFrontResponse(Response):
    """
    Response of queues.move_task_to_front endpoint.

    :param position: The new position of the task entry in the queue (index, -1
        represents bottom of queue)
    :type position: int
    """

    _service = "queues"
    _action = "move_task_to_front"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "position": {
                "description": "The new position of the task entry in the queue (index, -1 represents bottom of queue)",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, position=None, **kwargs):
        super(MoveTaskToFrontResponse, self).__init__(**kwargs)
        self.position = position

    @schema_property("position")
    def position(self):
        return self._property_position

    @position.setter
    def position(self, value):
        if value is None:
            self._property_position = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "position", six.integer_types)
        self._property_position = value


class PeekTaskRequest(Request):
    """
    Peek the next task from a given queue

    :param queue: ID of the queue
    :type queue: str
    """

    _service = "queues"
    _action = "peek_task"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {"queue": {"description": "ID of the queue", "type": "string"}},
        "required": ["queue"],
        "type": "object",
    }

    def __init__(self, queue, **kwargs):
        super(PeekTaskRequest, self).__init__(**kwargs)
        self.queue = queue

    @schema_property("queue")
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return

        self.assert_isinstance(value, "queue", six.string_types)
        self._property_queue = value


class PeekTaskResponse(Response):
    """
    Response of queues.peek_task endpoint.

    :param task: Task ID
    :type task: str
    """

    _service = "queues"
    _action = "peek_task"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {"task": {"description": "Task ID", "type": ["string", "null"]}},
        "type": "object",
    }

    def __init__(self, task=None, **kwargs):
        super(PeekTaskResponse, self).__init__(**kwargs)
        self.task = task

    @schema_property("task")
    def task(self):
        return self._property_task

    @task.setter
    def task(self, value):
        if value is None:
            self._property_task = None
            return

        self.assert_isinstance(value, "task", six.string_types)
        self._property_task = value


class RemoveTaskRequest(Request):
    """
    Removes a task entry from the queue.

    :param queue: Queue id
    :type queue: str
    :param task: Task id
    :type task: str
    """

    _service = "queues"
    _action = "remove_task"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "queue": {"description": "Queue id", "type": "string"},
            "task": {"description": "Task id", "type": "string"},
        },
        "required": ["queue", "task"],
        "type": "object",
    }

    def __init__(self, queue, task, **kwargs):
        super(RemoveTaskRequest, self).__init__(**kwargs)
        self.queue = queue
        self.task = task

    @schema_property("queue")
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return

        self.assert_isinstance(value, "queue", six.string_types)
        self._property_queue = value

    @schema_property("task")
    def task(self):
        return self._property_task

    @task.setter
    def task(self, value):
        if value is None:
            self._property_task = None
            return

        self.assert_isinstance(value, "task", six.string_types)
        self._property_task = value


class RemoveTaskResponse(Response):
    """
    Response of queues.remove_task endpoint.

    :param removed: Number of tasks removed (0 or 1)
    :type removed: int
    """

    _service = "queues"
    _action = "remove_task"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "removed": {
                "description": "Number of tasks removed (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, removed=None, **kwargs):
        super(RemoveTaskResponse, self).__init__(**kwargs)
        self.removed = removed

    @schema_property("removed")
    def removed(self):
        return self._property_removed

    @removed.setter
    def removed(self, value):
        if value is None:
            self._property_removed = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "removed", six.integer_types)
        self._property_removed = value


class UpdateRequest(Request):
    """
    Update queue information

    :param queue: Queue id
    :type queue: str
    :param name: Queue name Unique within the company.
    :type name: str
    :param tags: User-defined tags list
    :type tags: Sequence[str]
    :param system_tags: System tags list. This field is reserved for system use,
        please don't use it.
    :type system_tags: Sequence[str]
    """

    _service = "queues"
    _action = "update"
    _version = "2.23"
    _schema = {
        "definitions": {},
        "properties": {
            "name": {
                "description": "Queue name Unique within the company.",
                "type": "string",
            },
            "queue": {"description": "Queue id", "type": "string"},
            "system_tags": {
                "description": "System tags list. This field is reserved for system use, please don't use it.",
                "items": {"type": "string"},
                "type": "array",
            },
            "tags": {
                "description": "User-defined tags list",
                "items": {"type": "string"},
                "type": "array",
            },
        },
        "required": ["queue"],
        "type": "object",
    }

    def __init__(self, queue, name=None, tags=None, system_tags=None, **kwargs):
        super(UpdateRequest, self).__init__(**kwargs)
        self.queue = queue
        self.name = name
        self.tags = tags
        self.system_tags = system_tags

    @schema_property("queue")
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return

        self.assert_isinstance(value, "queue", six.string_types)
        self._property_queue = value

    @schema_property("name")
    def name(self):
        return self._property_name

    @name.setter
    def name(self, value):
        if value is None:
            self._property_name = None
            return

        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property("tags")
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

    @schema_property("system_tags")
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


class UpdateResponse(Response):
    """
    Response of queues.update endpoint.

    :param updated: Number of queues updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """

    _service = "queues"
    _action = "update"
    _version = "2.23"

    _schema = {
        "definitions": {},
        "properties": {
            "fields": {
                "additionalProperties": True,
                "description": "Updated fields names and values",
                "type": ["object", "null"],
            },
            "updated": {
                "description": "Number of queues updated (0 or 1)",
                "enum": [0, 1],
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, updated=None, fields=None, **kwargs):
        super(UpdateResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property("updated")
    def updated(self):
        return self._property_updated

    @updated.setter
    def updated(self, value):
        if value is None:
            self._property_updated = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "updated", six.integer_types)
        self._property_updated = value

    @schema_property("fields")
    def fields(self):
        return self._property_fields

    @fields.setter
    def fields(self, value):
        if value is None:
            self._property_fields = None
            return

        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


response_mapping = {
    GetByIdRequest: GetByIdResponse,
    GetAllRequest: GetAllResponse,
    GetDefaultRequest: GetDefaultResponse,
    CreateRequest: CreateResponse,
    UpdateRequest: UpdateResponse,
    DeleteRequest: DeleteResponse,
    AddTaskRequest: AddTaskResponse,
    GetNextTaskRequest: GetNextTaskResponse,
    RemoveTaskRequest: RemoveTaskResponse,
    MoveTaskForwardRequest: MoveTaskForwardResponse,
    MoveTaskBackwardRequest: MoveTaskBackwardResponse,
    MoveTaskToFrontRequest: MoveTaskToFrontResponse,
    MoveTaskToBackRequest: MoveTaskToBackResponse,
    GetQueueMetricsRequest: GetQueueMetricsResponse,
    AddOrUpdateMetadataRequest: AddOrUpdateMetadataResponse,
    DeleteMetadataRequest: DeleteMetadataResponse,
    PeekTaskRequest: PeekTaskResponse,
    GetNumEntriesRequest: GetNumEntriesResponse,
}
