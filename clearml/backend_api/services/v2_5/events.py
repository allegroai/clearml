"""
events service

Provides an API for running tasks to report events collected by the system.
"""
import enum

import six

from ....backend_api.session import BatchRequest, CompoundRequest, NonStrictDataModel, Request, Response, \
    schema_property, StringEnum


class MetricsScalarEvent(NonStrictDataModel):
    """
    Used for reporting scalar metrics during training task

    :param timestamp: Epoch milliseconds UTC, will be set by the server if not set.
    :type timestamp: float
    :param task: Task ID (required)
    :type task: str
    :param iter: Iteration
    :type iter: int
    :param metric: Metric name, e.g. 'count', 'loss', 'accuracy'
    :type metric: str
    :param variant: E.g. 'class_1', 'total', 'average
    :type variant: str
    :param value:
    :type value: float
    """
    _schema = {
        'description': 'Used for reporting scalar metrics during training task',
        'properties': {
            'iter': {'description': 'Iteration', 'type': 'integer'},
            'metric': {
                'description': "Metric name, e.g. 'count', 'loss', 'accuracy'",
                'type': 'string',
            },
            'task': {'description': 'Task ID (required)', 'type': 'string'},
            'timestamp': {
                'description': 'Epoch milliseconds UTC, will be set by the server if not set.',
                'type': ['number', 'null'],
            },
            'type': {
                'const': 'training_stats_scalar',
                'description': 'training_stats_vector',
            },
            'value': {'description': '', 'type': 'number'},
            'variant': {
                'description': "E.g. 'class_1', 'total', 'average",
                'type': 'string',
            },
        },
        'required': ['task', 'type'],
        'type': 'object',
    }

    def __init__(
            self, task, timestamp=None, iter=None, metric=None, variant=None, value=None, **kwargs):
        super(MetricsScalarEvent, self).__init__(**kwargs)
        self.timestamp = timestamp
        self.task = task
        self.iter = iter
        self.metric = metric
        self.variant = variant
        self.value = value

    @schema_property('timestamp')
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return

        self.assert_isinstance(value, "timestamp", six.integer_types + (float,))
        self._property_timestamp = value

    @schema_property('type')
    def type(self):
        return "training_stats_scalar"

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

    @schema_property('iter')
    def iter(self):
        return self._property_iter

    @iter.setter
    def iter(self, value):
        if value is None:
            self._property_iter = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "iter", six.integer_types)
        self._property_iter = value

    @schema_property('metric')
    def metric(self):
        return self._property_metric

    @metric.setter
    def metric(self, value):
        if value is None:
            self._property_metric = None
            return

        self.assert_isinstance(value, "metric", six.string_types)
        self._property_metric = value

    @schema_property('variant')
    def variant(self):
        return self._property_variant

    @variant.setter
    def variant(self, value):
        if value is None:
            self._property_variant = None
            return

        self.assert_isinstance(value, "variant", six.string_types)
        self._property_variant = value

    @schema_property('value')
    def value(self):
        return self._property_value

    @value.setter
    def value(self, value):
        if value is None:
            self._property_value = None
            return

        self.assert_isinstance(value, "value", six.integer_types + (float,))
        self._property_value = value


class MetricsVectorEvent(NonStrictDataModel):
    """
    Used for reporting vector metrics during training task

    :param timestamp: Epoch milliseconds UTC, will be set by the server if not set.
    :type timestamp: float
    :param task: Task ID (required)
    :type task: str
    :param iter: Iteration
    :type iter: int
    :param metric: Metric name, e.g. 'count', 'loss', 'accuracy'
    :type metric: str
    :param variant: E.g. 'class_1', 'total', 'average
    :type variant: str
    :param values: vector of float values
    :type values: Sequence[float]
    """
    _schema = {
        'description': 'Used for reporting vector metrics during training task',
        'properties': {
            'iter': {'description': 'Iteration', 'type': 'integer'},
            'metric': {
                'description': "Metric name, e.g. 'count', 'loss', 'accuracy'",
                'type': 'string',
            },
            'task': {'description': 'Task ID (required)', 'type': 'string'},
            'timestamp': {
                'description': 'Epoch milliseconds UTC, will be set by the server if not set.',
                'type': ['number', 'null'],
            },
            'type': {
                'const': 'training_stats_vector',
                'description': 'training_stats_vector',
            },
            'values': {
                'description': 'vector of float values',
                'items': {'type': 'number'},
                'type': 'array',
            },
            'variant': {
                'description': "E.g. 'class_1', 'total', 'average",
                'type': 'string',
            },
        },
        'required': ['task'],
        'type': 'object',
    }

    def __init__(
            self, task, timestamp=None, iter=None, metric=None, variant=None, values=None, **kwargs):
        super(MetricsVectorEvent, self).__init__(**kwargs)
        self.timestamp = timestamp
        self.task = task
        self.iter = iter
        self.metric = metric
        self.variant = variant
        self.values = values

    @schema_property('timestamp')
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return

        self.assert_isinstance(value, "timestamp", six.integer_types + (float,))
        self._property_timestamp = value

    @schema_property('type')
    def type(self):
        return "training_stats_vector"

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

    @schema_property('iter')
    def iter(self):
        return self._property_iter

    @iter.setter
    def iter(self, value):
        if value is None:
            self._property_iter = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "iter", six.integer_types)
        self._property_iter = value

    @schema_property('metric')
    def metric(self):
        return self._property_metric

    @metric.setter
    def metric(self, value):
        if value is None:
            self._property_metric = None
            return

        self.assert_isinstance(value, "metric", six.string_types)
        self._property_metric = value

    @schema_property('variant')
    def variant(self):
        return self._property_variant

    @variant.setter
    def variant(self, value):
        if value is None:
            self._property_variant = None
            return

        self.assert_isinstance(value, "variant", six.string_types)
        self._property_variant = value

    @schema_property('values')
    def values(self):
        return self._property_values

    @values.setter
    def values(self, value):
        if value is None:
            self._property_values = None
            return

        self.assert_isinstance(value, "values", (list, tuple))

        self.assert_isinstance(value, "values", six.integer_types + (float,), is_array=True)
        self._property_values = value


class MetricsImageEvent(NonStrictDataModel):
    """
    An image or video was dumped to storage for debugging

    :param timestamp: Epoch milliseconds UTC, will be set by the server if not set.
    :type timestamp: float
    :param task: Task ID (required)
    :type task: str
    :param iter: Iteration
    :type iter: int
    :param metric: Metric name, e.g. 'count', 'loss', 'accuracy'
    :type metric: str
    :param variant: E.g. 'class_1', 'total', 'average
    :type variant: str
    :param key: File key
    :type key: str
    :param url: File URL
    :type url: str
    """
    _schema = {
        'description': 'An image or video was dumped to storage for debugging',
        'properties': {
            'iter': {'description': 'Iteration', 'type': 'integer'},
            'key': {'description': 'File key', 'type': 'string'},
            'metric': {
                'description': "Metric name, e.g. 'count', 'loss', 'accuracy'",
                'type': 'string',
            },
            'task': {'description': 'Task ID (required)', 'type': 'string'},
            'timestamp': {
                'description': 'Epoch milliseconds UTC, will be set by the server if not set.',
                'type': ['number', 'null'],
            },
            'type': {'const': 'training_debug_image', 'description': ''},
            'url': {'description': 'File URL', 'type': 'string'},
            'variant': {
                'description': "E.g. 'class_1', 'total', 'average",
                'type': 'string',
            },
        },
        'required': ['task', 'type'],
        'type': 'object',
    }

    def __init__(
            self, task, timestamp=None, iter=None, metric=None, variant=None, key=None, url=None, **kwargs):
        super(MetricsImageEvent, self).__init__(**kwargs)
        self.timestamp = timestamp
        self.task = task
        self.iter = iter
        self.metric = metric
        self.variant = variant
        self.key = key
        self.url = url

    @schema_property('timestamp')
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return

        self.assert_isinstance(value, "timestamp", six.integer_types + (float,))
        self._property_timestamp = value

    @schema_property('type')
    def type(self):
        return "training_debug_image"

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

    @schema_property('iter')
    def iter(self):
        return self._property_iter

    @iter.setter
    def iter(self, value):
        if value is None:
            self._property_iter = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "iter", six.integer_types)
        self._property_iter = value

    @schema_property('metric')
    def metric(self):
        return self._property_metric

    @metric.setter
    def metric(self, value):
        if value is None:
            self._property_metric = None
            return

        self.assert_isinstance(value, "metric", six.string_types)
        self._property_metric = value

    @schema_property('variant')
    def variant(self):
        return self._property_variant

    @variant.setter
    def variant(self, value):
        if value is None:
            self._property_variant = None
            return

        self.assert_isinstance(value, "variant", six.string_types)
        self._property_variant = value

    @schema_property('key')
    def key(self):
        return self._property_key

    @key.setter
    def key(self, value):
        if value is None:
            self._property_key = None
            return

        self.assert_isinstance(value, "key", six.string_types)
        self._property_key = value

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


class MetricsPlotEvent(NonStrictDataModel):
    """
     An entire plot (not single datapoint) and it's layout.
                Used for plotting ROC curves, confidence matrices, etc. when evaluating the net.

    :param timestamp: Epoch milliseconds UTC, will be set by the server if not set.
    :type timestamp: float
    :param task: Task ID (required)
    :type task: str
    :param iter: Iteration
    :type iter: int
    :param metric: Metric name, e.g. 'count', 'loss', 'accuracy'
    :type metric: str
    :param variant: E.g. 'class_1', 'total', 'average
    :type variant: str
    :param plot_str: An entire plot (not single datapoint) and it's layout. Used
        for plotting ROC curves, confidence matrices, etc. when evaluating the net.
    :type plot_str: str
    """
    _schema = {
        'description': " An entire plot (not single datapoint) and it's layout.\n            Used for plotting ROC curves, confidence matrices, etc. when evaluating the net.",
        'properties': {
            'iter': {'description': 'Iteration', 'type': 'integer'},
            'metric': {
                'description': "Metric name, e.g. 'count', 'loss', 'accuracy'",
                'type': 'string',
            },
            'plot_str': {
                'description': "An entire plot (not single datapoint) and it's layout.\n                    Used for plotting ROC curves, confidence matrices, etc. when evaluating the net.\n                    ",
                'type': 'string',
            },
            'task': {'description': 'Task ID (required)', 'type': 'string'},
            'timestamp': {
                'description': 'Epoch milliseconds UTC, will be set by the server if not set.',
                'type': ['number', 'null'],
            },
            'type': {'const': 'plot', 'description': "'plot'"},
            'variant': {
                'description': "E.g. 'class_1', 'total', 'average",
                'type': 'string',
            },
        },
        'required': ['task', 'type'],
        'type': 'object',
    }

    def __init__(
            self, task, timestamp=None, iter=None, metric=None, variant=None, plot_str=None, **kwargs):
        super(MetricsPlotEvent, self).__init__(**kwargs)
        self.timestamp = timestamp
        self.task = task
        self.iter = iter
        self.metric = metric
        self.variant = variant
        self.plot_str = plot_str

    @schema_property('timestamp')
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return

        self.assert_isinstance(value, "timestamp", six.integer_types + (float,))
        self._property_timestamp = value

    @schema_property('type')
    def type(self):
        return "plot"

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

    @schema_property('iter')
    def iter(self):
        return self._property_iter

    @iter.setter
    def iter(self, value):
        if value is None:
            self._property_iter = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "iter", six.integer_types)
        self._property_iter = value

    @schema_property('metric')
    def metric(self):
        return self._property_metric

    @metric.setter
    def metric(self, value):
        if value is None:
            self._property_metric = None
            return

        self.assert_isinstance(value, "metric", six.string_types)
        self._property_metric = value

    @schema_property('variant')
    def variant(self):
        return self._property_variant

    @variant.setter
    def variant(self, value):
        if value is None:
            self._property_variant = None
            return

        self.assert_isinstance(value, "variant", six.string_types)
        self._property_variant = value

    @schema_property('plot_str')
    def plot_str(self):
        return self._property_plot_str

    @plot_str.setter
    def plot_str(self, value):
        if value is None:
            self._property_plot_str = None
            return

        self.assert_isinstance(value, "plot_str", six.string_types)
        self._property_plot_str = value


class ScalarKeyEnum(StringEnum):
    iter = "iter"
    timestamp = "timestamp"
    iso_time = "iso_time"


class LogLevelEnum(StringEnum):
    notset = "notset"
    debug = "debug"
    verbose = "verbose"
    info = "info"
    warn = "warn"
    warning = "warning"
    error = "error"
    fatal = "fatal"
    critical = "critical"


class TaskLogEvent(NonStrictDataModel):
    """
    A log event associated with a task.

    :param timestamp: Epoch milliseconds UTC, will be set by the server if not set.
    :type timestamp: float
    :param task: Task ID (required)
    :type task: str
    :param level: Log level.
    :type level: LogLevelEnum
    :param worker: Name of machine running the task.
    :type worker: str
    :param msg: Log message.
    :type msg: str
    """
    _schema = {
        'description': 'A log event associated with a task.',
        'properties': {
            'level': {
                '$ref': '#/definitions/log_level_enum',
                'description': 'Log level.',
            },
            'msg': {'description': 'Log message.', 'type': 'string'},
            'task': {'description': 'Task ID (required)', 'type': 'string'},
            'timestamp': {
                'description': 'Epoch milliseconds UTC, will be set by the server if not set.',
                'type': ['number', 'null'],
            },
            'type': {'const': 'log', 'description': "'log'"},
            'worker': {
                'description': 'Name of machine running the task.',
                'type': 'string',
            },
        },
        'required': ['task', 'type'],
        'type': 'object',
    }

    def __init__(
            self, task, timestamp=None, level=None, worker=None, msg=None, **kwargs):
        super(TaskLogEvent, self).__init__(**kwargs)
        self.timestamp = timestamp
        self.task = task
        self.level = level
        self.worker = worker
        self.msg = msg

    @schema_property('timestamp')
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return

        self.assert_isinstance(value, "timestamp", six.integer_types + (float,))
        self._property_timestamp = value

    @schema_property('type')
    def type(self):
        return "log"

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

    @schema_property('level')
    def level(self):
        return self._property_level

    @level.setter
    def level(self, value):
        if value is None:
            self._property_level = None
            return
        if isinstance(value, six.string_types):
            try:
                value = LogLevelEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "level", enum.Enum)
        self._property_level = value

    @schema_property('worker')
    def worker(self):
        return self._property_worker

    @worker.setter
    def worker(self, value):
        if value is None:
            self._property_worker = None
            return

        self.assert_isinstance(value, "worker", six.string_types)
        self._property_worker = value

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


class AddRequest(CompoundRequest):
    """
    Adds a single event

    """

    _service = "events"
    _action = "add"
    _version = "2.1"
    _item_prop_name = "event"
    _schema = {
        'anyOf': [
            {'$ref': '#/definitions/metrics_scalar_event'},
            {'$ref': '#/definitions/metrics_vector_event'},
            {'$ref': '#/definitions/metrics_image_event'},
            {'$ref': '#/definitions/metrics_plot_event'},
            {'$ref': '#/definitions/task_log_event'},
        ],
        'definitions': {
            'log_level_enum': {
                'enum': [
                    'notset',
                    'debug',
                    'verbose',
                    'info',
                    'warn',
                    'warning',
                    'error',
                    'fatal',
                    'critical',
                ],
                'type': 'string',
            },
            'metrics_image_event': {
                'description': 'An image or video was dumped to storage for debugging',
                'properties': {
                    'iter': {'description': 'Iteration', 'type': 'integer'},
                    'key': {'description': 'File key', 'type': 'string'},
                    'metric': {
                        'description': "Metric name, e.g. 'count', 'loss', 'accuracy'",
                        'type': 'string',
                    },
                    'task': {
                        'description': 'Task ID (required)',
                        'type': 'string',
                    },
                    'timestamp': {
                        'description': 'Epoch milliseconds UTC, will be set by the server if not set.',
                        'type': ['number', 'null'],
                    },
                    'type': {'const': 'training_debug_image', 'description': ''},
                    'url': {'description': 'File URL', 'type': 'string'},
                    'variant': {
                        'description': "E.g. 'class_1', 'total', 'average",
                        'type': 'string',
                    },
                },
                'required': ['task', 'type'],
                'type': 'object',
            },
            'metrics_plot_event': {
                'description': " An entire plot (not single datapoint) and it's layout.\n            Used for plotting ROC curves, confidence matrices, etc. when evaluating the net.",
                'properties': {
                    'iter': {'description': 'Iteration', 'type': 'integer'},
                    'metric': {
                        'description': "Metric name, e.g. 'count', 'loss', 'accuracy'",
                        'type': 'string',
                    },
                    'plot_str': {
                        'description': "An entire plot (not single datapoint) and it's layout.\n                    Used for plotting ROC curves, confidence matrices, etc. when evaluating the net.\n                    ",
                        'type': 'string',
                    },
                    'task': {
                        'description': 'Task ID (required)',
                        'type': 'string',
                    },
                    'timestamp': {
                        'description': 'Epoch milliseconds UTC, will be set by the server if not set.',
                        'type': ['number', 'null'],
                    },
                    'type': {'const': 'plot', 'description': "'plot'"},
                    'variant': {
                        'description': "E.g. 'class_1', 'total', 'average",
                        'type': 'string',
                    },
                },
                'required': ['task', 'type'],
                'type': 'object',
            },
            'metrics_scalar_event': {
                'description': 'Used for reporting scalar metrics during training task',
                'properties': {
                    'iter': {'description': 'Iteration', 'type': 'integer'},
                    'metric': {
                        'description': "Metric name, e.g. 'count', 'loss', 'accuracy'",
                        'type': 'string',
                    },
                    'task': {
                        'description': 'Task ID (required)',
                        'type': 'string',
                    },
                    'timestamp': {
                        'description': 'Epoch milliseconds UTC, will be set by the server if not set.',
                        'type': ['number', 'null'],
                    },
                    'type': {
                        'const': 'training_stats_scalar',
                        'description': 'training_stats_vector',
                    },
                    'value': {'description': '', 'type': 'number'},
                    'variant': {
                        'description': "E.g. 'class_1', 'total', 'average",
                        'type': 'string',
                    },
                },
                'required': ['task', 'type'],
                'type': 'object',
            },
            'metrics_vector_event': {
                'description': 'Used for reporting vector metrics during training task',
                'properties': {
                    'iter': {'description': 'Iteration', 'type': 'integer'},
                    'metric': {
                        'description': "Metric name, e.g. 'count', 'loss', 'accuracy'",
                        'type': 'string',
                    },
                    'task': {
                        'description': 'Task ID (required)',
                        'type': 'string',
                    },
                    'timestamp': {
                        'description': 'Epoch milliseconds UTC, will be set by the server if not set.',
                        'type': ['number', 'null'],
                    },
                    'type': {
                        'const': 'training_stats_vector',
                        'description': 'training_stats_vector',
                    },
                    'values': {
                        'description': 'vector of float values',
                        'items': {'type': 'number'},
                        'type': 'array',
                    },
                    'variant': {
                        'description': "E.g. 'class_1', 'total', 'average",
                        'type': 'string',
                    },
                },
                'required': ['task'],
                'type': 'object',
            },
            'task_log_event': {
                'description': 'A log event associated with a task.',
                'properties': {
                    'level': {
                        '$ref': '#/definitions/log_level_enum',
                        'description': 'Log level.',
                    },
                    'msg': {'description': 'Log message.', 'type': 'string'},
                    'task': {
                        'description': 'Task ID (required)',
                        'type': 'string',
                    },
                    'timestamp': {
                        'description': 'Epoch milliseconds UTC, will be set by the server if not set.',
                        'type': ['number', 'null'],
                    },
                    'type': {'const': 'log', 'description': "'log'"},
                    'worker': {
                        'description': 'Name of machine running the task.',
                        'type': 'string',
                    },
                },
                'required': ['task', 'type'],
                'type': 'object',
            },
        },
        'type': 'object',
    }

    def __init__(self, event):
        super(AddRequest, self).__init__()
        self.event = event

    @property
    def event(self):
        return self._property_event

    @event.setter
    def event(self, value):
        self.assert_isinstance(value, "event", (MetricsScalarEvent, MetricsVectorEvent, MetricsImageEvent, MetricsPlotEvent, TaskLogEvent))
        self._property_event = value


class AddResponse(Response):
    """
    Response of events.add endpoint.

    """
    _service = "events"
    _action = "add"
    _version = "2.1"

    _schema = {'additionalProperties': True, 'definitions': {}, 'type': 'object'}


class AddBatchRequest(BatchRequest):
    """
    Adds a batch of events in a single call (json-lines format, stream-friendly)

    """

    _service = "events"
    _action = "add_batch"
    _version = "2.1"
    _batched_request_cls = AddRequest


class AddBatchResponse(Response):
    """
    Response of events.add_batch endpoint.

    :param added:
    :type added: int
    :param errors:
    :type errors: int
    """
    _service = "events"
    _action = "add_batch"
    _version = "2.1"

    _schema = {
        'definitions': {},
        'properties': {
            'added': {'type': ['integer', 'null']},
            'errors': {'type': ['integer', 'null']},
        },
        'type': 'object',
    }

    def __init__(
            self, added=None, errors=None, **kwargs):
        super(AddBatchResponse, self).__init__(**kwargs)
        self.added = added
        self.errors = errors

    @schema_property('added')
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

    @schema_property('errors')
    def errors(self):
        return self._property_errors

    @errors.setter
    def errors(self, value):
        if value is None:
            self._property_errors = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "errors", six.integer_types)
        self._property_errors = value


class DebugImagesRequest(Request):
    """
    Get all debug images of a task

    :param task: Task ID
    :type task: str
    :param iters: Max number of latest iterations for which to return debug images
    :type iters: int
    :param scroll_id: Scroll ID of previous call (used for getting more results)
    :type scroll_id: str
    """

    _service = "events"
    _action = "debug_images"
    _version = "2.1"
    _schema = {
        'definitions': {},
        'properties': {
            'iters': {
                'description': 'Max number of latest iterations for which to return debug images',
                'type': 'integer',
            },
            'scroll_id': {
                'description': 'Scroll ID of previous call (used for getting more results)',
                'type': 'string',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }

    def __init__(
            self, task, iters=None, scroll_id=None, **kwargs):
        super(DebugImagesRequest, self).__init__(**kwargs)
        self.task = task
        self.iters = iters
        self.scroll_id = scroll_id

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

    @schema_property('iters')
    def iters(self):
        return self._property_iters

    @iters.setter
    def iters(self, value):
        if value is None:
            self._property_iters = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "iters", six.integer_types)
        self._property_iters = value

    @schema_property('scroll_id')
    def scroll_id(self):
        return self._property_scroll_id

    @scroll_id.setter
    def scroll_id(self, value):
        if value is None:
            self._property_scroll_id = None
            return

        self.assert_isinstance(value, "scroll_id", six.string_types)
        self._property_scroll_id = value


class DebugImagesResponse(Response):
    """
    Response of events.debug_images endpoint.

    :param task: Task ID
    :type task: str
    :param images: Images list
    :type images: Sequence[dict]
    :param returned: Number of results returned
    :type returned: int
    :param total: Total number of results available for this query
    :type total: float
    :param scroll_id: Scroll ID for getting more results
    :type scroll_id: str
    """
    _service = "events"
    _action = "debug_images"
    _version = "2.1"

    _schema = {
        'definitions': {},
        'properties': {
            'images': {
                'description': 'Images list',
                'items': {'type': 'object'},
                'type': ['array', 'null'],
            },
            'returned': {
                'description': 'Number of results returned',
                'type': ['integer', 'null'],
            },
            'scroll_id': {
                'description': 'Scroll ID for getting more results',
                'type': ['string', 'null'],
            },
            'task': {'description': 'Task ID', 'type': ['string', 'null']},
            'total': {
                'description': 'Total number of results available for this query',
                'type': ['number', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, task=None, images=None, returned=None, total=None, scroll_id=None, **kwargs):
        super(DebugImagesResponse, self).__init__(**kwargs)
        self.task = task
        self.images = images
        self.returned = returned
        self.total = total
        self.scroll_id = scroll_id

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

    @schema_property('images')
    def images(self):
        return self._property_images

    @images.setter
    def images(self, value):
        if value is None:
            self._property_images = None
            return

        self.assert_isinstance(value, "images", (list, tuple))

        self.assert_isinstance(value, "images", (dict,), is_array=True)
        self._property_images = value

    @schema_property('returned')
    def returned(self):
        return self._property_returned

    @returned.setter
    def returned(self, value):
        if value is None:
            self._property_returned = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "returned", six.integer_types)
        self._property_returned = value

    @schema_property('total')
    def total(self):
        return self._property_total

    @total.setter
    def total(self, value):
        if value is None:
            self._property_total = None
            return

        self.assert_isinstance(value, "total", six.integer_types + (float,))
        self._property_total = value

    @schema_property('scroll_id')
    def scroll_id(self):
        return self._property_scroll_id

    @scroll_id.setter
    def scroll_id(self, value):
        if value is None:
            self._property_scroll_id = None
            return

        self.assert_isinstance(value, "scroll_id", six.string_types)
        self._property_scroll_id = value


class DeleteForTaskRequest(Request):
    """
    Delete all task event. *This cannot be undone!*

    :param task: Task ID
    :type task: str
    :param allow_locked: Allow deleting events even if the task is locked
    :type allow_locked: bool
    """

    _service = "events"
    _action = "delete_for_task"
    _version = "2.1"
    _schema = {
        'definitions': {},
        'properties': {
            'allow_locked': {
                'default': False,
                'description': 'Allow deleting events even if the task is locked',
                'type': 'boolean',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }

    def __init__(
            self, task, allow_locked=False, **kwargs):
        super(DeleteForTaskRequest, self).__init__(**kwargs)
        self.task = task
        self.allow_locked = allow_locked

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

    @schema_property('allow_locked')
    def allow_locked(self):
        return self._property_allow_locked

    @allow_locked.setter
    def allow_locked(self, value):
        if value is None:
            self._property_allow_locked = None
            return

        self.assert_isinstance(value, "allow_locked", (bool,))
        self._property_allow_locked = value


class DeleteForTaskResponse(Response):
    """
    Response of events.delete_for_task endpoint.

    :param deleted: Number of deleted events
    :type deleted: bool
    """
    _service = "events"
    _action = "delete_for_task"
    _version = "2.1"

    _schema = {
        'definitions': {},
        'properties': {
            'deleted': {
                'description': 'Number of deleted events',
                'type': ['boolean', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, deleted=None, **kwargs):
        super(DeleteForTaskResponse, self).__init__(**kwargs)
        self.deleted = deleted

    @schema_property('deleted')
    def deleted(self):
        return self._property_deleted

    @deleted.setter
    def deleted(self, value):
        if value is None:
            self._property_deleted = None
            return

        self.assert_isinstance(value, "deleted", (bool,))
        self._property_deleted = value


class DownloadTaskLogRequest(Request):
    """
    Get an attachment containing the task's log

    :param task: Task ID
    :type task: str
    :param line_type: Line format type
    :type line_type: str
    :param line_format: Line string format. Used if the line type is 'text'
    :type line_format: str
    """

    _service = "events"
    _action = "download_task_log"
    _version = "2.1"
    _schema = {
        'definitions': {},
        'properties': {
            'line_format': {
                'default': '{asctime} {worker} {level} {msg}',
                'description': "Line string format. Used if the line type is 'text'",
                'type': 'string',
            },
            'line_type': {
                'description': 'Line format type',
                'enum': ['json', 'text'],
                'type': 'string',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }

    def __init__(
            self, task, line_type=None, line_format="{asctime} {worker} {level} {msg}", **kwargs):
        super(DownloadTaskLogRequest, self).__init__(**kwargs)
        self.task = task
        self.line_type = line_type
        self.line_format = line_format

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

    @schema_property('line_type')
    def line_type(self):
        return self._property_line_type

    @line_type.setter
    def line_type(self, value):
        if value is None:
            self._property_line_type = None
            return

        self.assert_isinstance(value, "line_type", six.string_types)
        self._property_line_type = value

    @schema_property('line_format')
    def line_format(self):
        return self._property_line_format

    @line_format.setter
    def line_format(self, value):
        if value is None:
            self._property_line_format = None
            return

        self.assert_isinstance(value, "line_format", six.string_types)
        self._property_line_format = value


class DownloadTaskLogResponse(Response):
    """
    Response of events.download_task_log endpoint.

    """
    _service = "events"
    _action = "download_task_log"
    _version = "2.1"

    _schema = {'definitions': {}, 'type': 'string'}


class GetMultiTaskPlotsRequest(Request):
    """
    Get 'plot' events for the given tasks

    :param tasks: List of task IDs
    :type tasks: Sequence[str]
    :param iters: Max number of latest iterations for which to return debug images
    :type iters: int
    :param scroll_id: Scroll ID of previous call (used for getting more results)
    :type scroll_id: str
    """

    _service = "events"
    _action = "get_multi_task_plots"
    _version = "2.1"
    _schema = {
        'definitions': {},
        'properties': {
            'iters': {
                'description': 'Max number of latest iterations for which to return debug images',
                'type': 'integer',
            },
            'scroll_id': {
                'description': 'Scroll ID of previous call (used for getting more results)',
                'type': 'string',
            },
            'tasks': {
                'description': 'List of task IDs',
                'items': {'description': 'Task ID', 'type': 'string'},
                'type': 'array',
            },
        },
        'required': ['tasks'],
        'type': 'object',
    }

    def __init__(
            self, tasks, iters=None, scroll_id=None, **kwargs):
        super(GetMultiTaskPlotsRequest, self).__init__(**kwargs)
        self.tasks = tasks
        self.iters = iters
        self.scroll_id = scroll_id

    @schema_property('tasks')
    def tasks(self):
        return self._property_tasks

    @tasks.setter
    def tasks(self, value):
        if value is None:
            self._property_tasks = None
            return

        self.assert_isinstance(value, "tasks", (list, tuple))

        self.assert_isinstance(value, "tasks", six.string_types, is_array=True)
        self._property_tasks = value

    @schema_property('iters')
    def iters(self):
        return self._property_iters

    @iters.setter
    def iters(self, value):
        if value is None:
            self._property_iters = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "iters", six.integer_types)
        self._property_iters = value

    @schema_property('scroll_id')
    def scroll_id(self):
        return self._property_scroll_id

    @scroll_id.setter
    def scroll_id(self, value):
        if value is None:
            self._property_scroll_id = None
            return

        self.assert_isinstance(value, "scroll_id", six.string_types)
        self._property_scroll_id = value


class GetMultiTaskPlotsResponse(Response):
    """
    Response of events.get_multi_task_plots endpoint.

    :param plots: Plots mapping (keyed by task name)
    :type plots: dict
    :param returned: Number of results returned
    :type returned: int
    :param total: Total number of results available for this query
    :type total: float
    :param scroll_id: Scroll ID for getting more results
    :type scroll_id: str
    """
    _service = "events"
    _action = "get_multi_task_plots"
    _version = "2.1"

    _schema = {
        'definitions': {},
        'properties': {
            'plots': {
                'description': 'Plots mapping (keyed by task name)',
                'type': ['object', 'null'],
            },
            'returned': {
                'description': 'Number of results returned',
                'type': ['integer', 'null'],
            },
            'scroll_id': {
                'description': 'Scroll ID for getting more results',
                'type': ['string', 'null'],
            },
            'total': {
                'description': 'Total number of results available for this query',
                'type': ['number', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, plots=None, returned=None, total=None, scroll_id=None, **kwargs):
        super(GetMultiTaskPlotsResponse, self).__init__(**kwargs)
        self.plots = plots
        self.returned = returned
        self.total = total
        self.scroll_id = scroll_id

    @schema_property('plots')
    def plots(self):
        return self._property_plots

    @plots.setter
    def plots(self, value):
        if value is None:
            self._property_plots = None
            return

        self.assert_isinstance(value, "plots", (dict,))
        self._property_plots = value

    @schema_property('returned')
    def returned(self):
        return self._property_returned

    @returned.setter
    def returned(self, value):
        if value is None:
            self._property_returned = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "returned", six.integer_types)
        self._property_returned = value

    @schema_property('total')
    def total(self):
        return self._property_total

    @total.setter
    def total(self, value):
        if value is None:
            self._property_total = None
            return

        self.assert_isinstance(value, "total", six.integer_types + (float,))
        self._property_total = value

    @schema_property('scroll_id')
    def scroll_id(self):
        return self._property_scroll_id

    @scroll_id.setter
    def scroll_id(self, value):
        if value is None:
            self._property_scroll_id = None
            return

        self.assert_isinstance(value, "scroll_id", six.string_types)
        self._property_scroll_id = value


class GetScalarMetricDataRequest(Request):
    """
    get scalar metric data for task

    :param task: task ID
    :type task: str
    :param metric: type of metric
    :type metric: str
    """

    _service = "events"
    _action = "get_scalar_metric_data"
    _version = "2.1"
    _schema = {
        'definitions': {},
        'properties': {
            'metric': {'description': 'type of metric', 'type': ['string', 'null']},
            'task': {'description': 'task ID', 'type': ['string', 'null']},
        },
        'type': 'object',
    }

    def __init__(
            self, task=None, metric=None, **kwargs):
        super(GetScalarMetricDataRequest, self).__init__(**kwargs)
        self.task = task
        self.metric = metric

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

    @schema_property('metric')
    def metric(self):
        return self._property_metric

    @metric.setter
    def metric(self, value):
        if value is None:
            self._property_metric = None
            return

        self.assert_isinstance(value, "metric", six.string_types)
        self._property_metric = value


class GetScalarMetricDataResponse(Response):
    """
    Response of events.get_scalar_metric_data endpoint.

    :param events: task scalar metric events
    :type events: Sequence[dict]
    :param returned: amount of events returned
    :type returned: int
    :param total: amount of events in task
    :type total: int
    :param scroll_id: Scroll ID of previous call (used for getting more results)
    :type scroll_id: str
    """
    _service = "events"
    _action = "get_scalar_metric_data"
    _version = "2.1"

    _schema = {
        'definitions': {},
        'properties': {
            'events': {
                'description': 'task scalar metric events',
                'items': {'type': 'object'},
                'type': ['array', 'null'],
            },
            'returned': {
                'description': 'amount of events returned',
                'type': ['integer', 'null'],
            },
            'scroll_id': {
                'description': 'Scroll ID of previous call (used for getting more results)',
                'type': ['string', 'null'],
            },
            'total': {
                'description': 'amount of events in task',
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, events=None, returned=None, total=None, scroll_id=None, **kwargs):
        super(GetScalarMetricDataResponse, self).__init__(**kwargs)
        self.events = events
        self.returned = returned
        self.total = total
        self.scroll_id = scroll_id

    @schema_property('events')
    def events(self):
        return self._property_events

    @events.setter
    def events(self, value):
        if value is None:
            self._property_events = None
            return

        self.assert_isinstance(value, "events", (list, tuple))

        self.assert_isinstance(value, "events", (dict,), is_array=True)
        self._property_events = value

    @schema_property('returned')
    def returned(self):
        return self._property_returned

    @returned.setter
    def returned(self, value):
        if value is None:
            self._property_returned = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "returned", six.integer_types)
        self._property_returned = value

    @schema_property('total')
    def total(self):
        return self._property_total

    @total.setter
    def total(self, value):
        if value is None:
            self._property_total = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "total", six.integer_types)
        self._property_total = value

    @schema_property('scroll_id')
    def scroll_id(self):
        return self._property_scroll_id

    @scroll_id.setter
    def scroll_id(self, value):
        if value is None:
            self._property_scroll_id = None
            return

        self.assert_isinstance(value, "scroll_id", six.string_types)
        self._property_scroll_id = value


class GetScalarMetricsAndVariantsRequest(Request):
    """
    get task scalar metrics and variants

    :param task: task ID
    :type task: str
    """

    _service = "events"
    _action = "get_scalar_metrics_and_variants"
    _version = "2.1"
    _schema = {
        'definitions': {},
        'properties': {'task': {'description': 'task ID', 'type': 'string'}},
        'required': ['task'],
        'type': 'object',
    }

    def __init__(
            self, task, **kwargs):
        super(GetScalarMetricsAndVariantsRequest, self).__init__(**kwargs)
        self.task = task

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


class GetScalarMetricsAndVariantsResponse(Response):
    """
    Response of events.get_scalar_metrics_and_variants endpoint.

    :param metrics:
    :type metrics: dict
    """
    _service = "events"
    _action = "get_scalar_metrics_and_variants"
    _version = "2.1"

    _schema = {
        'definitions': {},
        'properties': {
            'metrics': {'additionalProperties': True, 'type': ['object', 'null']},
        },
        'type': 'object',
    }

    def __init__(
            self, metrics=None, **kwargs):
        super(GetScalarMetricsAndVariantsResponse, self).__init__(**kwargs)
        self.metrics = metrics

    @schema_property('metrics')
    def metrics(self):
        return self._property_metrics

    @metrics.setter
    def metrics(self, value):
        if value is None:
            self._property_metrics = None
            return

        self.assert_isinstance(value, "metrics", (dict,))
        self._property_metrics = value


class GetTaskEventsRequest(Request):
    """
    Scroll through task events, sorted by timestamp

    :param task: Task ID
    :type task: str
    :param order: 'asc' (default) or 'desc'.
    :type order: str
    :param scroll_id: Pass this value on next call to get next page
    :type scroll_id: str
    :param batch_size: Number of events to return each time (default 500)
    :type batch_size: int
    :param event_type: Return only events of this type
    :type event_type: str
    """

    _service = "events"
    _action = "get_task_events"
    _version = "2.1"
    _schema = {
        'definitions': {},
        'properties': {
            'batch_size': {
                'description': 'Number of events to return each time (default 500)',
                'type': 'integer',
            },
            'event_type': {
                'description': 'Return only events of this type',
                'type': 'string',
            },
            'order': {
                'description': "'asc' (default) or 'desc'.",
                'enum': ['asc', 'desc'],
                'type': 'string',
            },
            'scroll_id': {
                'description': 'Pass this value on next call to get next page',
                'type': 'string',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }

    def __init__(
            self, task, order=None, scroll_id=None, batch_size=None, event_type=None, **kwargs):
        super(GetTaskEventsRequest, self).__init__(**kwargs)
        self.task = task
        self.order = order
        self.scroll_id = scroll_id
        self.batch_size = batch_size
        self.event_type = event_type

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

    @schema_property('order')
    def order(self):
        return self._property_order

    @order.setter
    def order(self, value):
        if value is None:
            self._property_order = None
            return

        self.assert_isinstance(value, "order", six.string_types)
        self._property_order = value

    @schema_property('scroll_id')
    def scroll_id(self):
        return self._property_scroll_id

    @scroll_id.setter
    def scroll_id(self, value):
        if value is None:
            self._property_scroll_id = None
            return

        self.assert_isinstance(value, "scroll_id", six.string_types)
        self._property_scroll_id = value

    @schema_property('batch_size')
    def batch_size(self):
        return self._property_batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value is None:
            self._property_batch_size = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "batch_size", six.integer_types)
        self._property_batch_size = value

    @schema_property('event_type')
    def event_type(self):
        return self._property_event_type

    @event_type.setter
    def event_type(self, value):
        if value is None:
            self._property_event_type = None
            return

        self.assert_isinstance(value, "event_type", six.string_types)
        self._property_event_type = value


class GetTaskEventsResponse(Response):
    """
    Response of events.get_task_events endpoint.

    :param events: Events list
    :type events: Sequence[dict]
    :param returned: Number of results returned
    :type returned: int
    :param total: Total number of results available for this query
    :type total: float
    :param scroll_id: Scroll ID for getting more results
    :type scroll_id: str
    """
    _service = "events"
    _action = "get_task_events"
    _version = "2.1"

    _schema = {
        'definitions': {},
        'properties': {
            'events': {
                'description': 'Events list',
                'items': {'type': 'object'},
                'type': ['array', 'null'],
            },
            'returned': {
                'description': 'Number of results returned',
                'type': ['integer', 'null'],
            },
            'scroll_id': {
                'description': 'Scroll ID for getting more results',
                'type': ['string', 'null'],
            },
            'total': {
                'description': 'Total number of results available for this query',
                'type': ['number', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, events=None, returned=None, total=None, scroll_id=None, **kwargs):
        super(GetTaskEventsResponse, self).__init__(**kwargs)
        self.events = events
        self.returned = returned
        self.total = total
        self.scroll_id = scroll_id

    @schema_property('events')
    def events(self):
        return self._property_events

    @events.setter
    def events(self, value):
        if value is None:
            self._property_events = None
            return

        self.assert_isinstance(value, "events", (list, tuple))

        self.assert_isinstance(value, "events", (dict,), is_array=True)
        self._property_events = value

    @schema_property('returned')
    def returned(self):
        return self._property_returned

    @returned.setter
    def returned(self, value):
        if value is None:
            self._property_returned = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "returned", six.integer_types)
        self._property_returned = value

    @schema_property('total')
    def total(self):
        return self._property_total

    @total.setter
    def total(self, value):
        if value is None:
            self._property_total = None
            return

        self.assert_isinstance(value, "total", six.integer_types + (float,))
        self._property_total = value

    @schema_property('scroll_id')
    def scroll_id(self):
        return self._property_scroll_id

    @scroll_id.setter
    def scroll_id(self, value):
        if value is None:
            self._property_scroll_id = None
            return

        self.assert_isinstance(value, "scroll_id", six.string_types)
        self._property_scroll_id = value


class GetTaskLatestScalarValuesRequest(Request):
    """
    Get the tasks's latest scalar values

    :param task: Task ID
    :type task: str
    """

    _service = "events"
    _action = "get_task_latest_scalar_values"
    _version = "2.1"
    _schema = {
        'definitions': {},
        'properties': {'task': {'description': 'Task ID', 'type': 'string'}},
        'required': ['task'],
        'type': 'object',
    }

    def __init__(
            self, task, **kwargs):
        super(GetTaskLatestScalarValuesRequest, self).__init__(**kwargs)
        self.task = task

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


class GetTaskLatestScalarValuesResponse(Response):
    """
    Response of events.get_task_latest_scalar_values endpoint.

    :param metrics:
    :type metrics: Sequence[dict]
    """
    _service = "events"
    _action = "get_task_latest_scalar_values"
    _version = "2.1"

    _schema = {
        'definitions': {},
        'properties': {
            'metrics': {
                'items': {
                    'properties': {
                        'name': {'description': 'Metric name', 'type': 'string'},
                        'variants': {
                            'items': {
                                'properties': {
                                    'last_100_value': {
                                        'description': 'Average of 100 last reported values',
                                        'type': 'number',
                                    },
                                    'last_value': {
                                        'description': 'Last reported value',
                                        'type': 'number',
                                    },
                                    'name': {
                                        'description': 'Variant name',
                                        'type': 'string',
                                    },
                                },
                                'type': 'object',
                            },
                            'type': 'array',
                        },
                    },
                    'type': 'object',
                },
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, metrics=None, **kwargs):
        super(GetTaskLatestScalarValuesResponse, self).__init__(**kwargs)
        self.metrics = metrics

    @schema_property('metrics')
    def metrics(self):
        return self._property_metrics

    @metrics.setter
    def metrics(self, value):
        if value is None:
            self._property_metrics = None
            return

        self.assert_isinstance(value, "metrics", (list, tuple))

        self.assert_isinstance(value, "metrics", (dict,), is_array=True)
        self._property_metrics = value


class GetTaskLogRequest(Request):
    """
    Get all 'log' events for this task

    :param task: Task ID
    :type task: str
    :param order: Timestamp order in which log events will be returned (defaults to
        ascending)
    :type order: str
    :param from: Where will the log entries be taken from (default to the head of
        the log)
    :type from: str
    :param scroll_id:
    :type scroll_id: str
    :param batch_size:
    :type batch_size: int
    """

    _service = "events"
    _action = "get_task_log"
    _version = "1.7"
    _schema = {
        'definitions': {},
        'properties': {
            'batch_size': {'description': '', 'type': 'integer'},
            'from': {
                'description': 'Where will the log entries be taken from (default to the head of the log)',
                'enum': ['head', 'tail'],
                'type': 'string',
            },
            'order': {
                'description': 'Timestamp order in which log events will be returned (defaults to ascending)',
                'enum': ['asc', 'desc'],
                'type': 'string',
            },
            'scroll_id': {'description': '', 'type': 'string'},
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }

    def __init__(
            self, task, order=None, from_=None, scroll_id=None, batch_size=None, **kwargs):
        super(GetTaskLogRequest, self).__init__(**kwargs)
        self.task = task
        self.order = order
        self.from_ = from_
        self.scroll_id = scroll_id
        self.batch_size = batch_size

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

    @schema_property('order')
    def order(self):
        return self._property_order

    @order.setter
    def order(self, value):
        if value is None:
            self._property_order = None
            return

        self.assert_isinstance(value, "order", six.string_types)
        self._property_order = value

    @schema_property('from')
    def from_(self):
        return self._property_from_

    @from_.setter
    def from_(self, value):
        if value is None:
            self._property_from_ = None
            return

        self.assert_isinstance(value, "from_", six.string_types)
        self._property_from_ = value

    @schema_property('scroll_id')
    def scroll_id(self):
        return self._property_scroll_id

    @scroll_id.setter
    def scroll_id(self, value):
        if value is None:
            self._property_scroll_id = None
            return

        self.assert_isinstance(value, "scroll_id", six.string_types)
        self._property_scroll_id = value

    @schema_property('batch_size')
    def batch_size(self):
        return self._property_batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value is None:
            self._property_batch_size = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "batch_size", six.integer_types)
        self._property_batch_size = value


class GetTaskLogResponse(Response):
    """
    Response of events.get_task_log endpoint.

    :param events: Log items list
    :type events: Sequence[dict]
    :param returned: Number of results returned
    :type returned: int
    :param total: Total number of results available for this query
    :type total: float
    :param scroll_id: Scroll ID for getting more results
    :type scroll_id: str
    """
    _service = "events"
    _action = "get_task_log"
    _version = "1.7"

    _schema = {
        'definitions': {},
        'properties': {
            'events': {
                'description': 'Log items list',
                'items': {'type': 'object'},
                'type': ['array', 'null'],
            },
            'returned': {
                'description': 'Number of results returned',
                'type': ['integer', 'null'],
            },
            'scroll_id': {
                'description': 'Scroll ID for getting more results',
                'type': ['string', 'null'],
            },
            'total': {
                'description': 'Total number of results available for this query',
                'type': ['number', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, events=None, returned=None, total=None, scroll_id=None, **kwargs):
        super(GetTaskLogResponse, self).__init__(**kwargs)
        self.events = events
        self.returned = returned
        self.total = total
        self.scroll_id = scroll_id

    @schema_property('events')
    def events(self):
        return self._property_events

    @events.setter
    def events(self, value):
        if value is None:
            self._property_events = None
            return

        self.assert_isinstance(value, "events", (list, tuple))

        self.assert_isinstance(value, "events", (dict,), is_array=True)
        self._property_events = value

    @schema_property('returned')
    def returned(self):
        return self._property_returned

    @returned.setter
    def returned(self, value):
        if value is None:
            self._property_returned = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "returned", six.integer_types)
        self._property_returned = value

    @schema_property('total')
    def total(self):
        return self._property_total

    @total.setter
    def total(self, value):
        if value is None:
            self._property_total = None
            return

        self.assert_isinstance(value, "total", six.integer_types + (float,))
        self._property_total = value

    @schema_property('scroll_id')
    def scroll_id(self):
        return self._property_scroll_id

    @scroll_id.setter
    def scroll_id(self, value):
        if value is None:
            self._property_scroll_id = None
            return

        self.assert_isinstance(value, "scroll_id", six.string_types)
        self._property_scroll_id = value


class GetTaskPlotsRequest(Request):
    """
    Get all 'plot' events for this task

    :param task: Task ID
    :type task: str
    :param iters: Max number of latest iterations for which to return debug images
    :type iters: int
    :param scroll_id: Scroll ID of previous call (used for getting more results)
    :type scroll_id: str
    """

    _service = "events"
    _action = "get_task_plots"
    _version = "2.1"
    _schema = {
        'definitions': {},
        'properties': {
            'iters': {
                'description': 'Max number of latest iterations for which to return debug images',
                'type': 'integer',
            },
            'scroll_id': {
                'description': 'Scroll ID of previous call (used for getting more results)',
                'type': 'string',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }

    def __init__(
            self, task, iters=None, scroll_id=None, **kwargs):
        super(GetTaskPlotsRequest, self).__init__(**kwargs)
        self.task = task
        self.iters = iters
        self.scroll_id = scroll_id

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

    @schema_property('iters')
    def iters(self):
        return self._property_iters

    @iters.setter
    def iters(self, value):
        if value is None:
            self._property_iters = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "iters", six.integer_types)
        self._property_iters = value

    @schema_property('scroll_id')
    def scroll_id(self):
        return self._property_scroll_id

    @scroll_id.setter
    def scroll_id(self, value):
        if value is None:
            self._property_scroll_id = None
            return

        self.assert_isinstance(value, "scroll_id", six.string_types)
        self._property_scroll_id = value


class GetTaskPlotsResponse(Response):
    """
    Response of events.get_task_plots endpoint.

    :param plots: Plots list
    :type plots: Sequence[dict]
    :param returned: Number of results returned
    :type returned: int
    :param total: Total number of results available for this query
    :type total: float
    :param scroll_id: Scroll ID for getting more results
    :type scroll_id: str
    """
    _service = "events"
    _action = "get_task_plots"
    _version = "2.1"

    _schema = {
        'definitions': {},
        'properties': {
            'plots': {
                'description': 'Plots list',
                'items': {'type': 'object'},
                'type': ['array', 'null'],
            },
            'returned': {
                'description': 'Number of results returned',
                'type': ['integer', 'null'],
            },
            'scroll_id': {
                'description': 'Scroll ID for getting more results',
                'type': ['string', 'null'],
            },
            'total': {
                'description': 'Total number of results available for this query',
                'type': ['number', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, plots=None, returned=None, total=None, scroll_id=None, **kwargs):
        super(GetTaskPlotsResponse, self).__init__(**kwargs)
        self.plots = plots
        self.returned = returned
        self.total = total
        self.scroll_id = scroll_id

    @schema_property('plots')
    def plots(self):
        return self._property_plots

    @plots.setter
    def plots(self, value):
        if value is None:
            self._property_plots = None
            return

        self.assert_isinstance(value, "plots", (list, tuple))

        self.assert_isinstance(value, "plots", (dict,), is_array=True)
        self._property_plots = value

    @schema_property('returned')
    def returned(self):
        return self._property_returned

    @returned.setter
    def returned(self, value):
        if value is None:
            self._property_returned = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "returned", six.integer_types)
        self._property_returned = value

    @schema_property('total')
    def total(self):
        return self._property_total

    @total.setter
    def total(self, value):
        if value is None:
            self._property_total = None
            return

        self.assert_isinstance(value, "total", six.integer_types + (float,))
        self._property_total = value

    @schema_property('scroll_id')
    def scroll_id(self):
        return self._property_scroll_id

    @scroll_id.setter
    def scroll_id(self, value):
        if value is None:
            self._property_scroll_id = None
            return

        self.assert_isinstance(value, "scroll_id", six.string_types)
        self._property_scroll_id = value


class GetVectorMetricsAndVariantsRequest(Request):
    """
    :param task: Task ID
    :type task: str
    """

    _service = "events"
    _action = "get_vector_metrics_and_variants"
    _version = "2.1"
    _schema = {
        'definitions': {},
        'properties': {'task': {'description': 'Task ID', 'type': 'string'}},
        'required': ['task'],
        'type': 'object',
    }

    def __init__(
            self, task, **kwargs):
        super(GetVectorMetricsAndVariantsRequest, self).__init__(**kwargs)
        self.task = task

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


class GetVectorMetricsAndVariantsResponse(Response):
    """
    Response of events.get_vector_metrics_and_variants endpoint.

    :param metrics:
    :type metrics: Sequence[dict]
    """
    _service = "events"
    _action = "get_vector_metrics_and_variants"
    _version = "2.1"

    _schema = {
        'definitions': {},
        'properties': {
            'metrics': {
                'description': '',
                'items': {'type': 'object'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, metrics=None, **kwargs):
        super(GetVectorMetricsAndVariantsResponse, self).__init__(**kwargs)
        self.metrics = metrics

    @schema_property('metrics')
    def metrics(self):
        return self._property_metrics

    @metrics.setter
    def metrics(self, value):
        if value is None:
            self._property_metrics = None
            return

        self.assert_isinstance(value, "metrics", (list, tuple))

        self.assert_isinstance(value, "metrics", (dict,), is_array=True)
        self._property_metrics = value


class MultiTaskScalarMetricsIterHistogramRequest(Request):
    """
    Used to compare scalar stats histogram of multiple tasks

    :param tasks: List of task Task IDs
    :type tasks: Sequence[str]
    :param samples: The amount of histogram points to return (0 to return all the
        points). Optional, the default value is 10000.
    :type samples: int
    :param key: Histogram x axis to use: iter - iteration number iso_time - event
        time as ISO formatted string timestamp - event timestamp as milliseconds since
        epoch
    :type key: ScalarKeyEnum
    """

    _service = "events"
    _action = "multi_task_scalar_metrics_iter_histogram"
    _version = "2.1"
    _schema = {
        'definitions': {
            'scalar_key_enum': {'enum': ['iter', 'timestamp', 'iso_time'], 'type': 'string'},
        },
        'properties': {
            'key': {
                '$ref': '#/definitions/scalar_key_enum',
                'description': '\n                        Histogram x axis to use:\n                        iter - iteration number\n                        iso_time - event time as ISO formatted string\n                        timestamp - event timestamp as milliseconds since epoch\n                        ',
            },
            'samples': {
                'description': 'The amount of histogram points to return (0 to return all the points). Optional, the default value is 10000.',
                'type': 'integer',
            },
            'tasks': {
                'description': 'List of task Task IDs',
                'items': {
                    'description': 'List of task Task IDs',
                    'type': 'string',
                },
                'type': 'array',
            },
        },
        'required': ['tasks'],
        'type': 'object',
    }

    def __init__(
            self, tasks, samples=None, key=None, **kwargs):
        super(MultiTaskScalarMetricsIterHistogramRequest, self).__init__(**kwargs)
        self.tasks = tasks
        self.samples = samples
        self.key = key

    @schema_property('tasks')
    def tasks(self):
        return self._property_tasks

    @tasks.setter
    def tasks(self, value):
        if value is None:
            self._property_tasks = None
            return

        self.assert_isinstance(value, "tasks", (list, tuple))

        self.assert_isinstance(value, "tasks", six.string_types, is_array=True)
        self._property_tasks = value

    @schema_property('samples')
    def samples(self):
        return self._property_samples

    @samples.setter
    def samples(self, value):
        if value is None:
            self._property_samples = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "samples", six.integer_types)
        self._property_samples = value

    @schema_property('key')
    def key(self):
        return self._property_key

    @key.setter
    def key(self, value):
        if value is None:
            self._property_key = None
            return
        if isinstance(value, six.string_types):
            try:
                value = ScalarKeyEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "key", enum.Enum)
        self._property_key = value


class MultiTaskScalarMetricsIterHistogramResponse(Response):
    """
    Response of events.multi_task_scalar_metrics_iter_histogram endpoint.

    """
    _service = "events"
    _action = "multi_task_scalar_metrics_iter_histogram"
    _version = "2.1"

    _schema = {'additionalProperties': True, 'definitions': {}, 'type': 'object'}


class ScalarMetricsIterHistogramRequest(Request):
    """
    Get histogram data of all the vector metrics and variants in the task

    :param task: Task ID
    :type task: str
    :param samples: The amount of histogram points to return (0 to return all the
        points). Optional, the default value is 10000.
    :type samples: int
    :param key: Histogram x axis to use: iter - iteration number iso_time - event
        time as ISO formatted string timestamp - event timestamp as milliseconds since
        epoch
    :type key: ScalarKeyEnum
    """

    _service = "events"
    _action = "scalar_metrics_iter_histogram"
    _version = "2.1"
    _schema = {
        'definitions': {
            'scalar_key_enum': {'enum': ['iter', 'timestamp', 'iso_time'], 'type': 'string'},
        },
        'properties': {
            'key': {
                '$ref': '#/definitions/scalar_key_enum',
                'description': '\n                        Histogram x axis to use:\n                        iter - iteration number\n                        iso_time - event time as ISO formatted string\n                        timestamp - event timestamp as milliseconds since epoch\n                        ',
            },
            'samples': {
                'description': 'The amount of histogram points to return (0 to return all the points). Optional, the default value is 10000.',
                'type': 'integer',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }

    def __init__(
            self, task, samples=None, key=None, **kwargs):
        super(ScalarMetricsIterHistogramRequest, self).__init__(**kwargs)
        self.task = task
        self.samples = samples
        self.key = key

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

    @schema_property('samples')
    def samples(self):
        return self._property_samples

    @samples.setter
    def samples(self, value):
        if value is None:
            self._property_samples = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "samples", six.integer_types)
        self._property_samples = value

    @schema_property('key')
    def key(self):
        return self._property_key

    @key.setter
    def key(self, value):
        if value is None:
            self._property_key = None
            return
        if isinstance(value, six.string_types):
            try:
                value = ScalarKeyEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "key", enum.Enum)
        self._property_key = value


class ScalarMetricsIterHistogramResponse(Response):
    """
    Response of events.scalar_metrics_iter_histogram endpoint.

    :param images:
    :type images: Sequence[dict]
    """
    _service = "events"
    _action = "scalar_metrics_iter_histogram"
    _version = "2.1"

    _schema = {
        'definitions': {},
        'properties': {
            'images': {'items': {'type': 'object'}, 'type': ['array', 'null']},
        },
        'type': 'object',
    }

    def __init__(
            self, images=None, **kwargs):
        super(ScalarMetricsIterHistogramResponse, self).__init__(**kwargs)
        self.images = images

    @schema_property('images')
    def images(self):
        return self._property_images

    @images.setter
    def images(self, value):
        if value is None:
            self._property_images = None
            return

        self.assert_isinstance(value, "images", (list, tuple))

        self.assert_isinstance(value, "images", (dict,), is_array=True)
        self._property_images = value


class VectorMetricsIterHistogramRequest(Request):
    """
    Get histogram data of all the scalar metrics and variants in the task

    :param task: Task ID
    :type task: str
    :param metric:
    :type metric: str
    :param variant:
    :type variant: str
    """

    _service = "events"
    _action = "vector_metrics_iter_histogram"
    _version = "2.1"
    _schema = {
        'definitions': {},
        'properties': {
            'metric': {'description': '', 'type': 'string'},
            'task': {'description': 'Task ID', 'type': 'string'},
            'variant': {'description': '', 'type': 'string'},
        },
        'required': ['task', 'metric', 'variant'],
        'type': 'object',
    }

    def __init__(
            self, task, metric, variant, **kwargs):
        super(VectorMetricsIterHistogramRequest, self).__init__(**kwargs)
        self.task = task
        self.metric = metric
        self.variant = variant

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

    @schema_property('metric')
    def metric(self):
        return self._property_metric

    @metric.setter
    def metric(self, value):
        if value is None:
            self._property_metric = None
            return

        self.assert_isinstance(value, "metric", six.string_types)
        self._property_metric = value

    @schema_property('variant')
    def variant(self):
        return self._property_variant

    @variant.setter
    def variant(self, value):
        if value is None:
            self._property_variant = None
            return

        self.assert_isinstance(value, "variant", six.string_types)
        self._property_variant = value


class VectorMetricsIterHistogramResponse(Response):
    """
    Response of events.vector_metrics_iter_histogram endpoint.

    :param images:
    :type images: Sequence[dict]
    """
    _service = "events"
    _action = "vector_metrics_iter_histogram"
    _version = "2.1"

    _schema = {
        'definitions': {},
        'properties': {
            'images': {'items': {'type': 'object'}, 'type': ['array', 'null']},
        },
        'type': 'object',
    }

    def __init__(
            self, images=None, **kwargs):
        super(VectorMetricsIterHistogramResponse, self).__init__(**kwargs)
        self.images = images

    @schema_property('images')
    def images(self):
        return self._property_images

    @images.setter
    def images(self, value):
        if value is None:
            self._property_images = None
            return

        self.assert_isinstance(value, "images", (list, tuple))

        self.assert_isinstance(value, "images", (dict,), is_array=True)
        self._property_images = value


response_mapping = {
    AddRequest: AddResponse,
    AddBatchRequest: AddBatchResponse,
    DeleteForTaskRequest: DeleteForTaskResponse,
    DebugImagesRequest: DebugImagesResponse,
    GetTaskLogRequest: GetTaskLogResponse,
    GetTaskEventsRequest: GetTaskEventsResponse,
    DownloadTaskLogRequest: DownloadTaskLogResponse,
    GetTaskPlotsRequest: GetTaskPlotsResponse,
    GetMultiTaskPlotsRequest: GetMultiTaskPlotsResponse,
    GetVectorMetricsAndVariantsRequest: GetVectorMetricsAndVariantsResponse,
    VectorMetricsIterHistogramRequest: VectorMetricsIterHistogramResponse,
    ScalarMetricsIterHistogramRequest: ScalarMetricsIterHistogramResponse,
    MultiTaskScalarMetricsIterHistogramRequest: MultiTaskScalarMetricsIterHistogramResponse,
    GetTaskLatestScalarValuesRequest: GetTaskLatestScalarValuesResponse,
    GetScalarMetricsAndVariantsRequest: GetScalarMetricsAndVariantsResponse,
    GetScalarMetricDataRequest: GetScalarMetricDataResponse,
}
