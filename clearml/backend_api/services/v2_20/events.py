"""
events service

Provides an API for running tasks to report events collected by the system.
"""
import six
import enum

from clearml.backend_api.session import (
    Request,
    BatchRequest,
    Response,
    NonStrictDataModel,
    CompoundRequest,
    schema_property,
    StringEnum,
)


class MetricVariants(NonStrictDataModel):
    """ """

    _schema = {
        "metric": {"description": "The metric name", "type": "string"},
        "type": "object",
        "variants": {
            "description": "The names of the metric variants",
            "items": {"type": "string"},
            "type": "array",
        },
    }


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
        "description": "Used for reporting scalar metrics during training task",
        "properties": {
            "iter": {"description": "Iteration", "type": "integer"},
            "metric": {
                "description": "Metric name, e.g. 'count', 'loss', 'accuracy'",
                "type": "string",
            },
            "task": {"description": "Task ID (required)", "type": "string"},
            "timestamp": {
                "description": "Epoch milliseconds UTC, will be set by the server if not set.",
                "type": ["number", "null"],
            },
            "type": {
                "const": "training_stats_scalar",
                "description": "training_stats_vector",
            },
            "value": {"description": "", "type": "number"},
            "variant": {
                "description": "E.g. 'class_1', 'total', 'average",
                "type": "string",
            },
        },
        "required": ["task", "type"],
        "type": "object",
    }

    def __init__(self, task, timestamp=None, iter=None, metric=None, variant=None, value=None, **kwargs):
        super(MetricsScalarEvent, self).__init__(**kwargs)
        self.timestamp = timestamp
        self.task = task
        self.iter = iter
        self.metric = metric
        self.variant = variant
        self.value = value

    @schema_property("timestamp")
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return

        self.assert_isinstance(value, "timestamp", six.integer_types + (float,))
        self._property_timestamp = value

    @schema_property("type")
    def type(self):
        return "training_stats_scalar"

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

    @schema_property("iter")
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

    @schema_property("metric")
    def metric(self):
        return self._property_metric

    @metric.setter
    def metric(self, value):
        if value is None:
            self._property_metric = None
            return

        self.assert_isinstance(value, "metric", six.string_types)
        self._property_metric = value

    @schema_property("variant")
    def variant(self):
        return self._property_variant

    @variant.setter
    def variant(self, value):
        if value is None:
            self._property_variant = None
            return

        self.assert_isinstance(value, "variant", six.string_types)
        self._property_variant = value

    @schema_property("value")
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
        "description": "Used for reporting vector metrics during training task",
        "properties": {
            "iter": {"description": "Iteration", "type": "integer"},
            "metric": {
                "description": "Metric name, e.g. 'count', 'loss', 'accuracy'",
                "type": "string",
            },
            "task": {"description": "Task ID (required)", "type": "string"},
            "timestamp": {
                "description": "Epoch milliseconds UTC, will be set by the server if not set.",
                "type": ["number", "null"],
            },
            "type": {
                "const": "training_stats_vector",
                "description": "training_stats_vector",
            },
            "values": {
                "description": "vector of float values",
                "items": {"type": "number"},
                "type": "array",
            },
            "variant": {
                "description": "E.g. 'class_1', 'total', 'average",
                "type": "string",
            },
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(self, task, timestamp=None, iter=None, metric=None, variant=None, values=None, **kwargs):
        super(MetricsVectorEvent, self).__init__(**kwargs)
        self.timestamp = timestamp
        self.task = task
        self.iter = iter
        self.metric = metric
        self.variant = variant
        self.values = values

    @schema_property("timestamp")
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return

        self.assert_isinstance(value, "timestamp", six.integer_types + (float,))
        self._property_timestamp = value

    @schema_property("type")
    def type(self):
        return "training_stats_vector"

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

    @schema_property("iter")
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

    @schema_property("metric")
    def metric(self):
        return self._property_metric

    @metric.setter
    def metric(self, value):
        if value is None:
            self._property_metric = None
            return

        self.assert_isinstance(value, "metric", six.string_types)
        self._property_metric = value

    @schema_property("variant")
    def variant(self):
        return self._property_variant

    @variant.setter
    def variant(self, value):
        if value is None:
            self._property_variant = None
            return

        self.assert_isinstance(value, "variant", six.string_types)
        self._property_variant = value

    @schema_property("values")
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
        "description": "An image or video was dumped to storage for debugging",
        "properties": {
            "iter": {"description": "Iteration", "type": "integer"},
            "key": {"description": "File key", "type": "string"},
            "metric": {
                "description": "Metric name, e.g. 'count', 'loss', 'accuracy'",
                "type": "string",
            },
            "task": {"description": "Task ID (required)", "type": "string"},
            "timestamp": {
                "description": "Epoch milliseconds UTC, will be set by the server if not set.",
                "type": ["number", "null"],
            },
            "type": {"const": "training_debug_image", "description": ""},
            "url": {"description": "File URL", "type": "string"},
            "variant": {
                "description": "E.g. 'class_1', 'total', 'average",
                "type": "string",
            },
        },
        "required": ["task", "type"],
        "type": "object",
    }

    def __init__(self, task, timestamp=None, iter=None, metric=None, variant=None, key=None, url=None, **kwargs):
        super(MetricsImageEvent, self).__init__(**kwargs)
        self.timestamp = timestamp
        self.task = task
        self.iter = iter
        self.metric = metric
        self.variant = variant
        self.key = key
        self.url = url

    @schema_property("timestamp")
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return

        self.assert_isinstance(value, "timestamp", six.integer_types + (float,))
        self._property_timestamp = value

    @schema_property("type")
    def type(self):
        return "training_debug_image"

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

    @schema_property("iter")
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

    @schema_property("metric")
    def metric(self):
        return self._property_metric

    @metric.setter
    def metric(self, value):
        if value is None:
            self._property_metric = None
            return

        self.assert_isinstance(value, "metric", six.string_types)
        self._property_metric = value

    @schema_property("variant")
    def variant(self):
        return self._property_variant

    @variant.setter
    def variant(self, value):
        if value is None:
            self._property_variant = None
            return

        self.assert_isinstance(value, "variant", six.string_types)
        self._property_variant = value

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

    @schema_property("url")
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
    :param skip_validation: If set then plot_str is not checked for a valid json.
        The default is False
    :type skip_validation: bool
    """

    _schema = {
        "description": (
            " An entire plot (not single datapoint) and it's layout. "
            " Used for plotting ROC curves, confidence matrices, etc. when evaluating the net."
        ),
        "properties": {
            "iter": {"description": "Iteration", "type": "integer"},
            "metric": {
                "description": "Metric name, e.g. 'count', 'loss', 'accuracy'",
                "type": "string",
            },
            "plot_str": {
                "description": (
                    "An entire plot (not single datapoint) and it's layout."
                    " Used for plotting ROC curves, confidence matrices, etc. when evaluating the net."
                ),
                "type": "string",
            },
            "skip_validation": {
                "description": "If set then plot_str is not checked for a valid json. The default is False",
                "type": "boolean",
            },
            "task": {"description": "Task ID (required)", "type": "string"},
            "timestamp": {
                "description": "Epoch milliseconds UTC, will be set by the server if not set.",
                "type": ["number", "null"],
            },
            "type": {"const": "plot", "description": "'plot'"},
            "variant": {
                "description": "E.g. 'class_1', 'total', 'average",
                "type": "string",
            },
        },
        "required": ["task", "type"],
        "type": "object",
    }

    def __init__(
        self, task, timestamp=None, iter=None, metric=None, variant=None, plot_str=None, skip_validation=None, **kwargs
    ):
        super(MetricsPlotEvent, self).__init__(**kwargs)
        self.timestamp = timestamp
        self.task = task
        self.iter = iter
        self.metric = metric
        self.variant = variant
        self.plot_str = plot_str
        self.skip_validation = skip_validation

    @schema_property("timestamp")
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return

        self.assert_isinstance(value, "timestamp", six.integer_types + (float,))
        self._property_timestamp = value

    @schema_property("type")
    def type(self):
        return "plot"

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

    @schema_property("iter")
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

    @schema_property("metric")
    def metric(self):
        return self._property_metric

    @metric.setter
    def metric(self, value):
        if value is None:
            self._property_metric = None
            return

        self.assert_isinstance(value, "metric", six.string_types)
        self._property_metric = value

    @schema_property("variant")
    def variant(self):
        return self._property_variant

    @variant.setter
    def variant(self, value):
        if value is None:
            self._property_variant = None
            return

        self.assert_isinstance(value, "variant", six.string_types)
        self._property_variant = value

    @schema_property("plot_str")
    def plot_str(self):
        return self._property_plot_str

    @plot_str.setter
    def plot_str(self, value):
        if value is None:
            self._property_plot_str = None
            return

        self.assert_isinstance(value, "plot_str", six.string_types)
        self._property_plot_str = value

    @schema_property("skip_validation")
    def skip_validation(self):
        return self._property_skip_validation

    @skip_validation.setter
    def skip_validation(self, value):
        if value is None:
            self._property_skip_validation = None
            return

        self.assert_isinstance(value, "skip_validation", (bool,))
        self._property_skip_validation = value


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


class EventTypeEnum(StringEnum):
    training_stats_scalar = "training_stats_scalar"
    training_stats_vector = "training_stats_vector"
    training_debug_image = "training_debug_image"
    plot = "plot"
    log = "log"


class TaskMetricVariants(NonStrictDataModel):
    """
    :param task: Task ID
    :type task: str
    :param metric: Metric name
    :type metric: str
    :param variants: Metric variant names
    :type variants: Sequence[str]
    """

    _schema = {
        "properties": {
            "metric": {"description": "Metric name", "type": "string"},
            "task": {"description": "Task ID", "type": "string"},
            "variants": {
                "description": "Metric variant names",
                "items": {"type": "string"},
                "type": "array",
            },
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(self, task, metric=None, variants=None, **kwargs):
        super(TaskMetricVariants, self).__init__(**kwargs)
        self.task = task
        self.metric = metric
        self.variants = variants

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

    @schema_property("metric")
    def metric(self):
        return self._property_metric

    @metric.setter
    def metric(self, value):
        if value is None:
            self._property_metric = None
            return

        self.assert_isinstance(value, "metric", six.string_types)
        self._property_metric = value

    @schema_property("variants")
    def variants(self):
        return self._property_variants

    @variants.setter
    def variants(self, value):
        if value is None:
            self._property_variants = None
            return

        self.assert_isinstance(value, "variants", (list, tuple))

        self.assert_isinstance(value, "variants", six.string_types, is_array=True)
        self._property_variants = value


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
        "description": "A log event associated with a task.",
        "properties": {
            "level": {
                "$ref": "#/definitions/log_level_enum",
                "description": "Log level.",
            },
            "msg": {"description": "Log message.", "type": "string"},
            "task": {"description": "Task ID (required)", "type": "string"},
            "timestamp": {
                "description": "Epoch milliseconds UTC, will be set by the server if not set.",
                "type": ["number", "null"],
            },
            "type": {"const": "log", "description": "'log'"},
            "worker": {
                "description": "Name of machine running the task.",
                "type": "string",
            },
        },
        "required": ["task", "type"],
        "type": "object",
    }

    def __init__(self, task, timestamp=None, level=None, worker=None, msg=None, **kwargs):
        super(TaskLogEvent, self).__init__(**kwargs)
        self.timestamp = timestamp
        self.task = task
        self.level = level
        self.worker = worker
        self.msg = msg

    @schema_property("timestamp")
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return

        self.assert_isinstance(value, "timestamp", six.integer_types + (float,))
        self._property_timestamp = value

    @schema_property("type")
    def type(self):
        return "log"

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

    @schema_property("level")
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

    @schema_property("worker")
    def worker(self):
        return self._property_worker

    @worker.setter
    def worker(self, value):
        if value is None:
            self._property_worker = None
            return

        self.assert_isinstance(value, "worker", six.string_types)
        self._property_worker = value

    @schema_property("msg")
    def msg(self):
        return self._property_msg

    @msg.setter
    def msg(self, value):
        if value is None:
            self._property_msg = None
            return

        self.assert_isinstance(value, "msg", six.string_types)
        self._property_msg = value


class DebugImagesResponseTaskMetrics(NonStrictDataModel):
    """
    :param task: Task ID
    :type task: str
    :param iterations:
    :type iterations: Sequence[dict]
    """

    _schema = {
        "properties": {
            "iterations": {
                "items": {
                    "properties": {
                        "events": {
                            "items": {
                                "description": "Debug image event",
                                "type": "object",
                            },
                            "type": "array",
                        },
                        "iter": {
                            "description": "Iteration number",
                            "type": "integer",
                        },
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            },
            "task": {"description": "Task ID", "type": ["string", "null"]},
        },
        "type": "object",
    }

    def __init__(self, task=None, iterations=None, **kwargs):
        super(DebugImagesResponseTaskMetrics, self).__init__(**kwargs)
        self.task = task
        self.iterations = iterations

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

    @schema_property("iterations")
    def iterations(self):
        return self._property_iterations

    @iterations.setter
    def iterations(self, value):
        if value is None:
            self._property_iterations = None
            return

        self.assert_isinstance(value, "iterations", (list, tuple))

        self.assert_isinstance(value, "iterations", (dict,), is_array=True)
        self._property_iterations = value


class DebugImagesResponse(Response):
    """
    :param scroll_id: Scroll ID for getting more results
    :type scroll_id: str
    :param metrics: Debug image events grouped by tasks and iterations
    :type metrics: Sequence[DebugImagesResponseTaskMetrics]
    """
    _service = "events"
    _action = "debug_images"
    _version = "2.20"

    _schema = {
        "properties": {
            "metrics": {
                "description": "Debug image events grouped by tasks and iterations",
                "items": {"$ref": "#/definitions/debug_images_response_task_metrics"},
                "type": ["array", "null"],
            },
            "scroll_id": {
                "description": "Scroll ID for getting more results",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, scroll_id=None, metrics=None, **kwargs):
        super(DebugImagesResponse, self).__init__(**kwargs)
        self.scroll_id = scroll_id
        self.metrics = metrics

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

    @schema_property("metrics")
    def metrics(self):
        return self._property_metrics

    @metrics.setter
    def metrics(self, value):
        if value is None:
            self._property_metrics = None
            return

        self.assert_isinstance(value, "metrics", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [DebugImagesResponseTaskMetrics.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "metrics", DebugImagesResponseTaskMetrics, is_array=True)
        self._property_metrics = value


class PlotsResponseTaskMetrics(NonStrictDataModel):
    """
    :param task: Task ID
    :type task: str
    :param iterations:
    :type iterations: Sequence[dict]
    """

    _schema = {
        "properties": {
            "iterations": {
                "items": {
                    "properties": {
                        "events": {
                            "items": {"description": "Plot event", "type": "object"},
                            "type": "array",
                        },
                        "iter": {
                            "description": "Iteration number",
                            "type": "integer",
                        },
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            },
            "task": {"description": "Task ID", "type": ["string", "null"]},
        },
        "type": "object",
    }

    def __init__(self, task=None, iterations=None, **kwargs):
        super(PlotsResponseTaskMetrics, self).__init__(**kwargs)
        self.task = task
        self.iterations = iterations

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

    @schema_property("iterations")
    def iterations(self):
        return self._property_iterations

    @iterations.setter
    def iterations(self, value):
        if value is None:
            self._property_iterations = None
            return

        self.assert_isinstance(value, "iterations", (list, tuple))

        self.assert_isinstance(value, "iterations", (dict,), is_array=True)
        self._property_iterations = value


class PlotsResponse(Response):
    """
    :param scroll_id: Scroll ID for getting more results
    :type scroll_id: str
    :param metrics: Plot events grouped by tasks and iterations
    :type metrics: Sequence[PlotsResponseTaskMetrics]
    """
    _service = "events"
    _action = "plots"
    _version = "2.20"

    _schema = {
        "properties": {
            "metrics": {
                "description": "Plot events grouped by tasks and iterations",
                "items": {"$ref": "#/definitions/plots_response_task_metrics"},
                "type": ["array", "null"],
            },
            "scroll_id": {
                "description": "Scroll ID for getting more results",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, scroll_id=None, metrics=None, **kwargs):
        super(PlotsResponse, self).__init__(**kwargs)
        self.scroll_id = scroll_id
        self.metrics = metrics

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

    @schema_property("metrics")
    def metrics(self):
        return self._property_metrics

    @metrics.setter
    def metrics(self, value):
        if value is None:
            self._property_metrics = None
            return

        self.assert_isinstance(value, "metrics", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [PlotsResponseTaskMetrics.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "metrics", PlotsResponseTaskMetrics, is_array=True)
        self._property_metrics = value


class DebugImageSampleResponse(NonStrictDataModel):
    """
    :param scroll_id: Scroll ID to pass to the next calls to get_debug_image_sample
        or next_debug_image_sample
    :type scroll_id: str
    :param event: Debug image event
    :type event: dict
    :param min_iteration: minimal valid iteration for the variant
    :type min_iteration: int
    :param max_iteration: maximal valid iteration for the variant
    :type max_iteration: int
    """

    _schema = {
        "properties": {
            "event": {"description": "Debug image event", "type": ["object", "null"]},
            "max_iteration": {
                "description": "maximal valid iteration for the variant",
                "type": ["integer", "null"],
            },
            "min_iteration": {
                "description": "minimal valid iteration for the variant",
                "type": ["integer", "null"],
            },
            "scroll_id": {
                "description": (
                    "Scroll ID to pass to the next calls to get_debug_image_sample or next_debug_image_sample"
                ),
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, scroll_id=None, event=None, min_iteration=None, max_iteration=None, **kwargs):
        super(DebugImageSampleResponse, self).__init__(**kwargs)
        self.scroll_id = scroll_id
        self.event = event
        self.min_iteration = min_iteration
        self.max_iteration = max_iteration

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

    @schema_property("event")
    def event(self):
        return self._property_event

    @event.setter
    def event(self, value):
        if value is None:
            self._property_event = None
            return

        self.assert_isinstance(value, "event", (dict,))
        self._property_event = value

    @schema_property("min_iteration")
    def min_iteration(self):
        return self._property_min_iteration

    @min_iteration.setter
    def min_iteration(self, value):
        if value is None:
            self._property_min_iteration = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "min_iteration", six.integer_types)
        self._property_min_iteration = value

    @schema_property("max_iteration")
    def max_iteration(self):
        return self._property_max_iteration

    @max_iteration.setter
    def max_iteration(self, value):
        if value is None:
            self._property_max_iteration = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "max_iteration", six.integer_types)
        self._property_max_iteration = value


class PlotSampleResponse(NonStrictDataModel):
    """
    :param scroll_id: Scroll ID to pass to the next calls to get_plot_sample or
        next_plot_sample
    :type scroll_id: str
    :param event: Plot event
    :type event: dict
    :param min_iteration: minimal valid iteration for the variant
    :type min_iteration: int
    :param max_iteration: maximal valid iteration for the variant
    :type max_iteration: int
    """

    _schema = {
        "properties": {
            "event": {"description": "Plot event", "type": ["object", "null"]},
            "max_iteration": {
                "description": "maximal valid iteration for the variant",
                "type": ["integer", "null"],
            },
            "min_iteration": {
                "description": "minimal valid iteration for the variant",
                "type": ["integer", "null"],
            },
            "scroll_id": {
                "description": "Scroll ID to pass to the next calls to get_plot_sample or next_plot_sample",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, scroll_id=None, event=None, min_iteration=None, max_iteration=None, **kwargs):
        super(PlotSampleResponse, self).__init__(**kwargs)
        self.scroll_id = scroll_id
        self.event = event
        self.min_iteration = min_iteration
        self.max_iteration = max_iteration

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

    @schema_property("event")
    def event(self):
        return self._property_event

    @event.setter
    def event(self, value):
        if value is None:
            self._property_event = None
            return

        self.assert_isinstance(value, "event", (dict,))
        self._property_event = value

    @schema_property("min_iteration")
    def min_iteration(self):
        return self._property_min_iteration

    @min_iteration.setter
    def min_iteration(self, value):
        if value is None:
            self._property_min_iteration = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "min_iteration", six.integer_types)
        self._property_min_iteration = value

    @schema_property("max_iteration")
    def max_iteration(self):
        return self._property_max_iteration

    @max_iteration.setter
    def max_iteration(self, value):
        if value is None:
            self._property_max_iteration = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "max_iteration", six.integer_types)
        self._property_max_iteration = value


class AddRequest(CompoundRequest):
    """
    Adds a single event

    """

    _service = "events"
    _action = "add"
    _version = "2.20"
    _item_prop_name = "event"
    _schema = {
        "anyOf": [
            {"$ref": "#/definitions/metrics_scalar_event"},
            {"$ref": "#/definitions/metrics_vector_event"},
            {"$ref": "#/definitions/metrics_image_event"},
            {"$ref": "#/definitions/metrics_plot_event"},
            {"$ref": "#/definitions/task_log_event"},
        ],
        "definitions": {
            "log_level_enum": {
                "enum": [
                    "notset",
                    "debug",
                    "verbose",
                    "info",
                    "warn",
                    "warning",
                    "error",
                    "fatal",
                    "critical",
                ],
                "type": "string",
            },
            "metrics_image_event": {
                "description": "An image or video was dumped to storage for debugging",
                "properties": {
                    "iter": {"description": "Iteration", "type": "integer"},
                    "key": {"description": "File key", "type": "string"},
                    "metric": {
                        "description": "Metric name, e.g. 'count', 'loss', 'accuracy'",
                        "type": "string",
                    },
                    "task": {
                        "description": "Task ID (required)",
                        "type": "string",
                    },
                    "timestamp": {
                        "description": "Epoch milliseconds UTC, will be set by the server if not set.",
                        "type": ["number", "null"],
                    },
                    "type": {"const": "training_debug_image", "description": ""},
                    "url": {"description": "File URL", "type": "string"},
                    "variant": {
                        "description": "E.g. 'class_1', 'total', 'average",
                        "type": "string",
                    },
                },
                "required": ["task", "type"],
                "type": "object",
            },
            "metrics_plot_event": {
                "description": (
                    " An entire plot (not single datapoint) and it's layout. "
                    "Used for plotting ROC curves, confidence matrices, etc. when evaluating the net."
                ),
                "properties": {
                    "iter": {"description": "Iteration", "type": "integer"},
                    "metric": {
                        "description": "Metric name, e.g. 'count', 'loss', 'accuracy'",
                        "type": "string",
                    },
                    "plot_str": {
                        "description": (
                            "An entire plot (not single datapoint) and it's layout. "
                            "Used for plotting ROC curves, confidence matrices, etc. "
                            "when evaluating the net."
                        ),
                        "type": "string",
                    },
                    "skip_validation": {
                        "description": "If set then plot_str is not checked for a valid json. The default is False",
                        "type": "boolean",
                    },
                    "task": {
                        "description": "Task ID (required)",
                        "type": "string",
                    },
                    "timestamp": {
                        "description": "Epoch milliseconds UTC, will be set by the server if not set.",
                        "type": ["number", "null"],
                    },
                    "type": {"const": "plot", "description": "'plot'"},
                    "variant": {
                        "description": "E.g. 'class_1', 'total', 'average",
                        "type": "string",
                    },
                },
                "required": ["task", "type"],
                "type": "object",
            },
            "metrics_scalar_event": {
                "description": "Used for reporting scalar metrics during training task",
                "properties": {
                    "iter": {"description": "Iteration", "type": "integer"},
                    "metric": {
                        "description": "Metric name, e.g. 'count', 'loss', 'accuracy'",
                        "type": "string",
                    },
                    "task": {
                        "description": "Task ID (required)",
                        "type": "string",
                    },
                    "timestamp": {
                        "description": "Epoch milliseconds UTC, will be set by the server if not set.",
                        "type": ["number", "null"],
                    },
                    "type": {
                        "const": "training_stats_scalar",
                        "description": "training_stats_vector",
                    },
                    "value": {"description": "", "type": "number"},
                    "variant": {
                        "description": "E.g. 'class_1', 'total', 'average",
                        "type": "string",
                    },
                },
                "required": ["task", "type"],
                "type": "object",
            },
            "metrics_vector_event": {
                "description": "Used for reporting vector metrics during training task",
                "properties": {
                    "iter": {"description": "Iteration", "type": "integer"},
                    "metric": {
                        "description": "Metric name, e.g. 'count', 'loss', 'accuracy'",
                        "type": "string",
                    },
                    "task": {
                        "description": "Task ID (required)",
                        "type": "string",
                    },
                    "timestamp": {
                        "description": "Epoch milliseconds UTC, will be set by the server if not set.",
                        "type": ["number", "null"],
                    },
                    "type": {
                        "const": "training_stats_vector",
                        "description": "training_stats_vector",
                    },
                    "values": {
                        "description": "vector of float values",
                        "items": {"type": "number"},
                        "type": "array",
                    },
                    "variant": {
                        "description": "E.g. 'class_1', 'total', 'average",
                        "type": "string",
                    },
                },
                "required": ["task"],
                "type": "object",
            },
            "task_log_event": {
                "description": "A log event associated with a task.",
                "properties": {
                    "level": {
                        "$ref": "#/definitions/log_level_enum",
                        "description": "Log level.",
                    },
                    "msg": {"description": "Log message.", "type": "string"},
                    "task": {
                        "description": "Task ID (required)",
                        "type": "string",
                    },
                    "timestamp": {
                        "description": "Epoch milliseconds UTC, will be set by the server if not set.",
                        "type": ["number", "null"],
                    },
                    "type": {"const": "log", "description": "'log'"},
                    "worker": {
                        "description": "Name of machine running the task.",
                        "type": "string",
                    },
                },
                "required": ["task", "type"],
                "type": "object",
            },
        },
        "type": "object",
    }

    def __init__(self, event):
        super(AddRequest, self).__init__()
        self.event = event

    @property
    def event(self):
        return self._property_event

    @event.setter
    def event(self, value):
        self.assert_isinstance(
            value,
            "event",
            (
                MetricsScalarEvent,
                MetricsVectorEvent,
                MetricsImageEvent,
                MetricsPlotEvent,
                TaskLogEvent,
            ),
        )
        self._property_event = value


class AddResponse(Response):
    """
    Response of events.add endpoint.

    """

    _service = "events"
    _action = "add"
    _version = "2.20"

    _schema = {"additionalProperties": True, "definitions": {}, "type": "object"}


class AddBatchRequest(BatchRequest):
    """
    Adds a batch of events in a single call (json-lines format, stream-friendly)

    """

    _service = "events"
    _action = "add_batch"
    _version = "2.20"
    _batched_request_cls = AddRequest


class AddBatchResponse(Response):
    """
    Response of events.add_batch endpoint.

    :param added:
    :type added: int
    :param errors:
    :type errors: int
    :param errors_info:
    :type errors_info: dict
    """

    _service = "events"
    _action = "add_batch"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "added": {"type": ["integer", "null"]},
            "errors": {"type": ["integer", "null"]},
            "errors_info": {"type": ["object", "null"]},
        },
        "type": "object",
    }

    def __init__(self, added=None, errors=None, errors_info=None, **kwargs):
        super(AddBatchResponse, self).__init__(**kwargs)
        self.added = added
        self.errors = errors
        self.errors_info = errors_info

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

    @schema_property("errors")
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

    @schema_property("errors_info")
    def errors_info(self):
        return self._property_errors_info

    @errors_info.setter
    def errors_info(self, value):
        if value is None:
            self._property_errors_info = None
            return

        self.assert_isinstance(value, "errors_info", (dict,))
        self._property_errors_info = value


class ClearScrollRequest(Request):
    """
    Clear an open Scroll ID

    :param scroll_id: Scroll ID as returned by previous events service calls
    :type scroll_id: str
    """

    _service = "events"
    _action = "clear_scroll"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "scroll_id": {
                "description": "Scroll ID as returned by previous events service calls",
                "type": "string",
            }
        },
        "required": ["scroll_id"],
        "type": "object",
    }

    def __init__(self, scroll_id, **kwargs):
        super(ClearScrollRequest, self).__init__(**kwargs)
        self.scroll_id = scroll_id

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


class ClearScrollResponse(Response):
    """
    Response of events.clear_scroll endpoint.

    """

    _service = "events"
    _action = "clear_scroll"
    _version = "2.20"

    _schema = {"additionalProperties": False, "definitions": {}, "type": "object"}


class ClearTaskLogRequest(Request):
    """
    Remove old logs from task

    :param task: Task ID
    :type task: str
    :param allow_locked: Allow deleting events even if the task is locked
    :type allow_locked: bool
    :param threshold_sec: The amount of seconds ago to retain the log records. The
        older log records will be deleted. If not passed or 0 then all the log records
        for the task will be deleted
    :type threshold_sec: int
    """

    _service = "events"
    _action = "clear_task_log"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "allow_locked": {
                "default": False,
                "description": "Allow deleting events even if the task is locked",
                "type": "boolean",
            },
            "task": {"description": "Task ID", "type": "string"},
            "threshold_sec": {
                "description": (
                    "The amount of seconds ago to retain the log records. "
                    "The older log records will be deleted. If not passed or 0 "
                    "then all the log records for the task will be deleted"
                ),
                "type": "integer",
            },
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(self, task, allow_locked=False, threshold_sec=None, **kwargs):
        super(ClearTaskLogRequest, self).__init__(**kwargs)
        self.task = task
        self.allow_locked = allow_locked
        self.threshold_sec = threshold_sec

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

    @schema_property("allow_locked")
    def allow_locked(self):
        return self._property_allow_locked

    @allow_locked.setter
    def allow_locked(self, value):
        if value is None:
            self._property_allow_locked = None
            return

        self.assert_isinstance(value, "allow_locked", (bool,))
        self._property_allow_locked = value

    @schema_property("threshold_sec")
    def threshold_sec(self):
        return self._property_threshold_sec

    @threshold_sec.setter
    def threshold_sec(self, value):
        if value is None:
            self._property_threshold_sec = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "threshold_sec", six.integer_types)
        self._property_threshold_sec = value


class ClearTaskLogResponse(Response):
    """
    Response of events.clear_task_log endpoint.

    :param deleted: The number of deleted log records
    :type deleted: int
    """

    _service = "events"
    _action = "clear_task_log"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "deleted": {
                "description": "The number of deleted log records",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, deleted=None, **kwargs):
        super(ClearTaskLogResponse, self).__init__(**kwargs)
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


class DebugImagesRequest(Request):
    """
    Get the debug image events for the requested amount of iterations per each task

    :param metrics: List of metrics and variants
    :type metrics: Sequence[TaskMetricVariants]
    :param iters: Max number of latest iterations for which to return debug images
    :type iters: int
    :param navigate_earlier: If set then events are retreived from latest
        iterations to earliest ones. Otherwise from earliest iterations to the latest.
        The default is True
    :type navigate_earlier: bool
    :param refresh: If set then scroll will be moved to the latest iterations. The
        default is False
    :type refresh: bool
    :param scroll_id: Scroll ID of previous call (used for getting more results)
    :type scroll_id: str
    """

    _service = "events"
    _action = "debug_images"
    _version = "2.20"
    _schema = {
        "definitions": {
            "task_metric_variants": {
                "properties": {
                    "metric": {"description": "Metric name", "type": "string"},
                    "task": {"description": "Task ID", "type": "string"},
                    "variants": {
                        "description": "Metric variant names",
                        "items": {"type": "string"},
                        "type": "array",
                    },
                },
                "required": ["task"],
                "type": "object",
            }
        },
        "properties": {
            "iters": {
                "description": "Max number of latest iterations for which to return debug images",
                "type": "integer",
            },
            "metrics": {
                "description": "List of metrics and variants",
                "items": {"$ref": "#/definitions/task_metric_variants"},
                "type": "array",
            },
            "navigate_earlier": {
                "description": (
                    "If set then events are retreived from latest "
                    "iterations to earliest ones. Otherwise from "
                    "earliest iterations to the latest. The default is True"
                ),
                "type": "boolean",
            },
            "refresh": {
                "description": "If set then scroll will be moved to the latest iterations. The default is False",
                "type": "boolean",
            },
            "scroll_id": {
                "description": "Scroll ID of previous call (used for getting more results)",
                "type": "string",
            },
        },
        "required": ["metrics"],
        "type": "object",
    }

    def __init__(self, metrics, iters=None, navigate_earlier=None, refresh=None, scroll_id=None, **kwargs):
        super(DebugImagesRequest, self).__init__(**kwargs)
        self.metrics = metrics
        self.iters = iters
        self.navigate_earlier = navigate_earlier
        self.refresh = refresh
        self.scroll_id = scroll_id

    @schema_property("metrics")
    def metrics(self):
        return self._property_metrics

    @metrics.setter
    def metrics(self, value):
        if value is None:
            self._property_metrics = None
            return

        self.assert_isinstance(value, "metrics", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [TaskMetricVariants.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "metrics", TaskMetricVariants, is_array=True)
        self._property_metrics = value

    @schema_property("iters")
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

    @schema_property("navigate_earlier")
    def navigate_earlier(self):
        return self._property_navigate_earlier

    @navigate_earlier.setter
    def navigate_earlier(self, value):
        if value is None:
            self._property_navigate_earlier = None
            return

        self.assert_isinstance(value, "navigate_earlier", (bool,))
        self._property_navigate_earlier = value

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
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "allow_locked": {
                "default": False,
                "description": "Allow deleting events even if the task is locked",
                "type": "boolean",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(self, task, allow_locked=False, **kwargs):
        super(DeleteForTaskRequest, self).__init__(**kwargs)
        self.task = task
        self.allow_locked = allow_locked

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

    @schema_property("allow_locked")
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
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "deleted": {
                "description": "Number of deleted events",
                "type": ["boolean", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, deleted=None, **kwargs):
        super(DeleteForTaskResponse, self).__init__(**kwargs)
        self.deleted = deleted

    @schema_property("deleted")
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
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "line_format": {
                "default": "{asctime} {worker} {level} {msg}",
                "description": "Line string format. Used if the line type is 'text'",
                "type": "string",
            },
            "line_type": {
                "description": "Line format type",
                "enum": ["json", "text"],
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(self, task, line_type=None, line_format="{asctime} {worker} {level} {msg}", **kwargs):
        super(DownloadTaskLogRequest, self).__init__(**kwargs)
        self.task = task
        self.line_type = line_type
        self.line_format = line_format

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

    @schema_property("line_type")
    def line_type(self):
        return self._property_line_type

    @line_type.setter
    def line_type(self, value):
        if value is None:
            self._property_line_type = None
            return

        self.assert_isinstance(value, "line_type", six.string_types)
        self._property_line_type = value

    @schema_property("line_format")
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
    _version = "2.20"

    _schema = {"definitions": {}, "type": "string"}


class GetDebugImageSampleRequest(Request):
    """
    Return the debug image per metric and variant for the provided iteration

    :param task: Task ID
    :type task: str
    :param metric: Metric name
    :type metric: str
    :param variant: Metric variant
    :type variant: str
    :param iteration: The iteration to bring debug image from. If not specified
        then the latest reported image is retrieved
    :type iteration: int
    :param refresh: If set then scroll state will be refreshed to reflect the
        latest changes in the debug images
    :type refresh: bool
    :param scroll_id: Scroll ID from the previous call to get_debug_image_sample or
        empty
    :type scroll_id: str
    :param navigate_current_metric: If set then subsequent navigation with
        next_debug_image_sample is done on the debug images for the passed metric only.
        Otherwise for all the metrics
    :type navigate_current_metric: bool
    """

    _service = "events"
    _action = "get_debug_image_sample"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "iteration": {
                "description": (
                    "The iteration to bring debug image from. If not specified "
                    "then the latest reported image is retrieved"
                ),
                "type": "integer",
            },
            "metric": {"description": "Metric name", "type": "string"},
            "navigate_current_metric": {
                "default": True,
                "description": (
                    "If set then subsequent navigation with "
                    "next_debug_image_sample is done on the debug images for "
                    "the passed metric only. Otherwise for all the metrics"
                ),
                "type": "boolean",
            },
            "refresh": {
                "description": (
                    "If set then scroll state will be refreshed to reflect the latest changes in the debug images"
                ),
                "type": "boolean",
            },
            "scroll_id": {
                "description": "Scroll ID from the previous call to get_debug_image_sample or empty",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
            "variant": {"description": "Metric variant", "type": "string"},
        },
        "required": ["task", "metric", "variant"],
        "type": "object",
    }

    def __init__(
        self,
        task,
        metric,
        variant,
        iteration=None,
        refresh=None,
        scroll_id=None,
        navigate_current_metric=True,
        **kwargs
    ):
        super(GetDebugImageSampleRequest, self).__init__(**kwargs)
        self.task = task
        self.metric = metric
        self.variant = variant
        self.iteration = iteration
        self.refresh = refresh
        self.scroll_id = scroll_id
        self.navigate_current_metric = navigate_current_metric

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

    @schema_property("metric")
    def metric(self):
        return self._property_metric

    @metric.setter
    def metric(self, value):
        if value is None:
            self._property_metric = None
            return

        self.assert_isinstance(value, "metric", six.string_types)
        self._property_metric = value

    @schema_property("variant")
    def variant(self):
        return self._property_variant

    @variant.setter
    def variant(self, value):
        if value is None:
            self._property_variant = None
            return

        self.assert_isinstance(value, "variant", six.string_types)
        self._property_variant = value

    @schema_property("iteration")
    def iteration(self):
        return self._property_iteration

    @iteration.setter
    def iteration(self, value):
        if value is None:
            self._property_iteration = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "iteration", six.integer_types)
        self._property_iteration = value

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

    @schema_property("navigate_current_metric")
    def navigate_current_metric(self):
        return self._property_navigate_current_metric

    @navigate_current_metric.setter
    def navigate_current_metric(self, value):
        if value is None:
            self._property_navigate_current_metric = None
            return

        self.assert_isinstance(value, "navigate_current_metric", (bool,))
        self._property_navigate_current_metric = value


class GetDebugImageSampleResponse(Response):
    """
    Response of events.get_debug_image_sample endpoint.

    """

    _service = "events"
    _action = "get_debug_image_sample"
    _version = "2.20"

    _schema = {
        "$ref": "#/definitions/debug_image_sample_response",
        "definitions": {
            "debug_image_sample_response": {
                "properties": {
                    "event": {
                        "description": "Debug image event",
                        "type": ["object", "null"],
                    },
                    "max_iteration": {
                        "description": "maximal valid iteration for the variant",
                        "type": ["integer", "null"],
                    },
                    "min_iteration": {
                        "description": "minimal valid iteration for the variant",
                        "type": ["integer", "null"],
                    },
                    "scroll_id": {
                        "description": (
                            "Scroll ID to pass to the next calls to get_debug_image_sample or next_debug_image_sample"
                        ),
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            }
        },
    }


class GetMultiTaskPlotsRequest(Request):
    """
    Get 'plot' events for the given tasks

    :param tasks: List of task IDs
    :type tasks: Sequence[str]
    :param iters: Max number of latest iterations for which to return debug images
    :type iters: int
    :param scroll_id: Scroll ID of previous call (used for getting more results)
    :type scroll_id: str
    :param no_scroll: If Truethen no scroll is created. Suitable for one time calls
    :type no_scroll: bool
    """

    _service = "events"
    _action = "get_multi_task_plots"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "iters": {
                "description": "Max number of latest iterations for which to return debug images",
                "type": "integer",
            },
            "no_scroll": {
                "default": False,
                "description": "If Truethen no scroll is created. Suitable for one time calls",
                "type": "boolean",
            },
            "scroll_id": {
                "description": "Scroll ID of previous call (used for getting more results)",
                "type": "string",
            },
            "tasks": {
                "description": "List of task IDs",
                "items": {"description": "Task ID", "type": "string"},
                "type": "array",
            },
        },
        "required": ["tasks"],
        "type": "object",
    }

    def __init__(self, tasks, iters=None, scroll_id=None, no_scroll=False, **kwargs):
        super(GetMultiTaskPlotsRequest, self).__init__(**kwargs)
        self.tasks = tasks
        self.iters = iters
        self.scroll_id = scroll_id
        self.no_scroll = no_scroll

    @schema_property("tasks")
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

    @schema_property("iters")
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

    @schema_property("no_scroll")
    def no_scroll(self):
        return self._property_no_scroll

    @no_scroll.setter
    def no_scroll(self, value):
        if value is None:
            self._property_no_scroll = None
            return

        self.assert_isinstance(value, "no_scroll", (bool,))
        self._property_no_scroll = value


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
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "plots": {
                "description": "Plots mapping (keyed by task name)",
                "type": ["object", "null"],
            },
            "returned": {
                "description": "Number of results returned",
                "type": ["integer", "null"],
            },
            "scroll_id": {
                "description": "Scroll ID for getting more results",
                "type": ["string", "null"],
            },
            "total": {
                "description": "Total number of results available for this query",
                "type": ["number", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, plots=None, returned=None, total=None, scroll_id=None, **kwargs):
        super(GetMultiTaskPlotsResponse, self).__init__(**kwargs)
        self.plots = plots
        self.returned = returned
        self.total = total
        self.scroll_id = scroll_id

    @schema_property("plots")
    def plots(self):
        return self._property_plots

    @plots.setter
    def plots(self, value):
        if value is None:
            self._property_plots = None
            return

        self.assert_isinstance(value, "plots", (dict,))
        self._property_plots = value

    @schema_property("returned")
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

    @schema_property("total")
    def total(self):
        return self._property_total

    @total.setter
    def total(self, value):
        if value is None:
            self._property_total = None
            return

        self.assert_isinstance(value, "total", six.integer_types + (float,))
        self._property_total = value

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


class GetPlotSampleRequest(Request):
    """
    Return the plot per metric and variant for the provided iteration

    :param task: Task ID
    :type task: str
    :param metric: Metric name
    :type metric: str
    :param variant: Metric variant
    :type variant: str
    :param iteration: The iteration to bring plot from. If not specified then the
        latest reported plot is retrieved
    :type iteration: int
    :param refresh: If set then scroll state will be refreshed to reflect the
        latest changes in the plots
    :type refresh: bool
    :param scroll_id: Scroll ID from the previous call to get_plot_sample or empty
    :type scroll_id: str
    :param navigate_current_metric: If set then subsequent navigation with
        next_plot_sample is done on the plots for the passed metric only. Otherwise for
        all the metrics
    :type navigate_current_metric: bool
    """

    _service = "events"
    _action = "get_plot_sample"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "iteration": {
                "description": (
                    "The iteration to bring plot from. If not specified then the latest reported plot is retrieved"
                ),
                "type": "integer",
            },
            "metric": {"description": "Metric name", "type": "string"},
            "navigate_current_metric": {
                "default": True,
                "description": (
                    "If set then subsequent navigation with next_plot_sample is done on the "
                    "plots for the passed metric only. Otherwise for all the metrics"
                ),
                "type": "boolean",
            },
            "refresh": {
                "description": "If set then scroll state will be refreshed to reflect the latest changes in the plots",
                "type": "boolean",
            },
            "scroll_id": {
                "description": "Scroll ID from the previous call to get_plot_sample or empty",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
            "variant": {"description": "Metric variant", "type": "string"},
        },
        "required": ["task", "metric", "variant"],
        "type": "object",
    }

    def __init__(
        self,
        task,
        metric,
        variant,
        iteration=None,
        refresh=None,
        scroll_id=None,
        navigate_current_metric=True,
        **kwargs
    ):
        super(GetPlotSampleRequest, self).__init__(**kwargs)
        self.task = task
        self.metric = metric
        self.variant = variant
        self.iteration = iteration
        self.refresh = refresh
        self.scroll_id = scroll_id
        self.navigate_current_metric = navigate_current_metric

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

    @schema_property("metric")
    def metric(self):
        return self._property_metric

    @metric.setter
    def metric(self, value):
        if value is None:
            self._property_metric = None
            return

        self.assert_isinstance(value, "metric", six.string_types)
        self._property_metric = value

    @schema_property("variant")
    def variant(self):
        return self._property_variant

    @variant.setter
    def variant(self, value):
        if value is None:
            self._property_variant = None
            return

        self.assert_isinstance(value, "variant", six.string_types)
        self._property_variant = value

    @schema_property("iteration")
    def iteration(self):
        return self._property_iteration

    @iteration.setter
    def iteration(self, value):
        if value is None:
            self._property_iteration = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "iteration", six.integer_types)
        self._property_iteration = value

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

    @schema_property("navigate_current_metric")
    def navigate_current_metric(self):
        return self._property_navigate_current_metric

    @navigate_current_metric.setter
    def navigate_current_metric(self, value):
        if value is None:
            self._property_navigate_current_metric = None
            return

        self.assert_isinstance(value, "navigate_current_metric", (bool,))
        self._property_navigate_current_metric = value


class GetPlotSampleResponse(Response):
    """
    Response of events.get_plot_sample endpoint.

    """

    _service = "events"
    _action = "get_plot_sample"
    _version = "2.20"

    _schema = {
        "$ref": "#/definitions/plot_sample_response",
        "definitions": {
            "plot_sample_response": {
                "properties": {
                    "event": {
                        "description": "Plot event",
                        "type": ["object", "null"],
                    },
                    "max_iteration": {
                        "description": "maximal valid iteration for the variant",
                        "type": ["integer", "null"],
                    },
                    "min_iteration": {
                        "description": "minimal valid iteration for the variant",
                        "type": ["integer", "null"],
                    },
                    "scroll_id": {
                        "description": "Scroll ID to pass to the next calls to get_plot_sample or next_plot_sample",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            }
        },
    }


class GetScalarMetricDataRequest(Request):
    """
    get scalar metric data for task

    :param task: task ID
    :type task: str
    :param metric: type of metric
    :type metric: str
    :param no_scroll: If Truethen no scroll is created. Suitable for one time calls
    :type no_scroll: bool
    """

    _service = "events"
    _action = "get_scalar_metric_data"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "metric": {"description": "type of metric", "type": ["string", "null"]},
            "no_scroll": {
                "default": False,
                "description": "If Truethen no scroll is created. Suitable for one time calls",
                "type": ["boolean", "null"],
            },
            "task": {"description": "task ID", "type": ["string", "null"]},
        },
        "type": "object",
    }

    def __init__(self, task=None, metric=None, no_scroll=False, **kwargs):
        super(GetScalarMetricDataRequest, self).__init__(**kwargs)
        self.task = task
        self.metric = metric
        self.no_scroll = no_scroll

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

    @schema_property("metric")
    def metric(self):
        return self._property_metric

    @metric.setter
    def metric(self, value):
        if value is None:
            self._property_metric = None
            return

        self.assert_isinstance(value, "metric", six.string_types)
        self._property_metric = value

    @schema_property("no_scroll")
    def no_scroll(self):
        return self._property_no_scroll

    @no_scroll.setter
    def no_scroll(self, value):
        if value is None:
            self._property_no_scroll = None
            return

        self.assert_isinstance(value, "no_scroll", (bool,))
        self._property_no_scroll = value


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
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "events": {
                "description": "task scalar metric events",
                "items": {"type": "object"},
                "type": ["array", "null"],
            },
            "returned": {
                "description": "amount of events returned",
                "type": ["integer", "null"],
            },
            "scroll_id": {
                "description": "Scroll ID of previous call (used for getting more results)",
                "type": ["string", "null"],
            },
            "total": {
                "description": "amount of events in task",
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, events=None, returned=None, total=None, scroll_id=None, **kwargs):
        super(GetScalarMetricDataResponse, self).__init__(**kwargs)
        self.events = events
        self.returned = returned
        self.total = total
        self.scroll_id = scroll_id

    @schema_property("events")
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

    @schema_property("returned")
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

    @schema_property("total")
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


class GetScalarMetricsAndVariantsRequest(Request):
    """
    get task scalar metrics and variants

    :param task: task ID
    :type task: str
    """

    _service = "events"
    _action = "get_scalar_metrics_and_variants"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {"task": {"description": "task ID", "type": "string"}},
        "required": ["task"],
        "type": "object",
    }

    def __init__(self, task, **kwargs):
        super(GetScalarMetricsAndVariantsRequest, self).__init__(**kwargs)
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


class GetScalarMetricsAndVariantsResponse(Response):
    """
    Response of events.get_scalar_metrics_and_variants endpoint.

    :param metrics:
    :type metrics: dict
    """

    _service = "events"
    _action = "get_scalar_metrics_and_variants"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {"metrics": {"additionalProperties": True, "type": ["object", "null"]}},
        "type": "object",
    }

    def __init__(self, metrics=None, **kwargs):
        super(GetScalarMetricsAndVariantsResponse, self).__init__(**kwargs)
        self.metrics = metrics

    @schema_property("metrics")
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
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "batch_size": {
                "description": "Number of events to return each time (default 500)",
                "type": "integer",
            },
            "event_type": {
                "description": "Return only events of this type",
                "type": "string",
            },
            "order": {
                "description": "'asc' (default) or 'desc'.",
                "enum": ["asc", "desc"],
                "type": "string",
            },
            "scroll_id": {
                "description": "Pass this value on next call to get next page",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(self, task, order=None, scroll_id=None, batch_size=None, event_type=None, **kwargs):
        super(GetTaskEventsRequest, self).__init__(**kwargs)
        self.task = task
        self.order = order
        self.scroll_id = scroll_id
        self.batch_size = batch_size
        self.event_type = event_type

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

    @schema_property("order")
    def order(self):
        return self._property_order

    @order.setter
    def order(self, value):
        if value is None:
            self._property_order = None
            return

        self.assert_isinstance(value, "order", six.string_types)
        self._property_order = value

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

    @schema_property("batch_size")
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

    @schema_property("event_type")
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
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "events": {
                "description": "Events list",
                "items": {"type": "object"},
                "type": ["array", "null"],
            },
            "returned": {
                "description": "Number of results returned",
                "type": ["integer", "null"],
            },
            "scroll_id": {
                "description": "Scroll ID for getting more results",
                "type": ["string", "null"],
            },
            "total": {
                "description": "Total number of results available for this query",
                "type": ["number", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, events=None, returned=None, total=None, scroll_id=None, **kwargs):
        super(GetTaskEventsResponse, self).__init__(**kwargs)
        self.events = events
        self.returned = returned
        self.total = total
        self.scroll_id = scroll_id

    @schema_property("events")
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

    @schema_property("returned")
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

    @schema_property("total")
    def total(self):
        return self._property_total

    @total.setter
    def total(self, value):
        if value is None:
            self._property_total = None
            return

        self.assert_isinstance(value, "total", six.integer_types + (float,))
        self._property_total = value

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


class GetTaskLatestScalarValuesRequest(Request):
    """
    Get the tasks's latest scalar values

    :param task: Task ID
    :type task: str
    """

    _service = "events"
    _action = "get_task_latest_scalar_values"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {"task": {"description": "Task ID", "type": "string"}},
        "required": ["task"],
        "type": "object",
    }

    def __init__(self, task, **kwargs):
        super(GetTaskLatestScalarValuesRequest, self).__init__(**kwargs)
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


class GetTaskLatestScalarValuesResponse(Response):
    """
    Response of events.get_task_latest_scalar_values endpoint.

    :param metrics:
    :type metrics: Sequence[dict]
    """

    _service = "events"
    _action = "get_task_latest_scalar_values"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "metrics": {
                "items": {
                    "properties": {
                        "name": {"description": "Metric name", "type": "string"},
                        "variants": {
                            "items": {
                                "properties": {
                                    "last_100_value": {
                                        "description": "Average of 100 last reported values",
                                        "type": "number",
                                    },
                                    "last_value": {
                                        "description": "Last reported value",
                                        "type": "number",
                                    },
                                    "name": {
                                        "description": "Variant name",
                                        "type": "string",
                                    },
                                },
                                "type": "object",
                            },
                            "type": "array",
                        },
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, metrics=None, **kwargs):
        super(GetTaskLatestScalarValuesResponse, self).__init__(**kwargs)
        self.metrics = metrics

    @schema_property("metrics")
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
    Get 'log' events for this task

    :param task: Task ID
    :type task: str
    :param batch_size: The amount of log events to return
    :type batch_size: int
    :param navigate_earlier: If set then log events are retreived from the latest
        to the earliest ones (in timestamp descending order, unless order='asc').
        Otherwise from the earliest to the latest ones (in timestamp ascending order,
        unless order='desc'). The default is True
    :type navigate_earlier: bool
    :param from_timestamp: Epoch time in UTC ms to use as the navigation start.
        Optional. If not provided, reference timestamp is determined by the
        'navigate_earlier' parameter (if true, reference timestamp is the last
        timestamp and if false, reference timestamp is the first timestamp)
    :type from_timestamp: float
    :param order: If set, changes the order in which log events are returned based
        on the value of 'navigate_earlier'
    :type order: str
    """

    _service = "events"
    _action = "get_task_log"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "batch_size": {
                "description": "The amount of log events to return",
                "type": "integer",
            },
            "from_timestamp": {
                "description": (
                    "Epoch time in UTC ms to use as the navigation start. Optional. If not provided, "
                    "reference timestamp is determined by the 'navigate_earlier' parameter (if true, "
                    "reference timestamp is the last timestamp and if false, reference timestamp "
                    "is the first timestamp)"
                ),
                "type": "number",
            },
            "navigate_earlier": {
                "description": (
                    "If set then log events are retreived from the latest to the earliest ones "
                    "(in timestamp descending order, unless order='asc'). Otherwise from the earliest to "
                    "the latest ones (in timestamp ascending order, unless order='desc'). "
                    "The default is True"
                ),
                "type": "boolean",
            },
            "order": {
                "description": (
                    "If set, changes the order in which log events are returned based on the value "
                    "of 'navigate_earlier'"
                ),
                "enum": ["asc", "desc"],
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(self, task, batch_size=None, navigate_earlier=None, from_timestamp=None, order=None, **kwargs):
        super(GetTaskLogRequest, self).__init__(**kwargs)
        self.task = task
        self.batch_size = batch_size
        self.navigate_earlier = navigate_earlier
        self.from_timestamp = from_timestamp
        self.order = order

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

    @schema_property("batch_size")
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

    @schema_property("navigate_earlier")
    def navigate_earlier(self):
        return self._property_navigate_earlier

    @navigate_earlier.setter
    def navigate_earlier(self, value):
        if value is None:
            self._property_navigate_earlier = None
            return

        self.assert_isinstance(value, "navigate_earlier", (bool,))
        self._property_navigate_earlier = value

    @schema_property("from_timestamp")
    def from_timestamp(self):
        return self._property_from_timestamp

    @from_timestamp.setter
    def from_timestamp(self, value):
        if value is None:
            self._property_from_timestamp = None
            return

        self.assert_isinstance(value, "from_timestamp", six.integer_types + (float,))
        self._property_from_timestamp = value

    @schema_property("order")
    def order(self):
        return self._property_order

    @order.setter
    def order(self, value):
        if value is None:
            self._property_order = None
            return

        self.assert_isinstance(value, "order", six.string_types)
        self._property_order = value


class GetTaskLogResponse(Response):
    """
    Response of events.get_task_log endpoint.

    :param events: Log items list
    :type events: Sequence[dict]
    :param returned: Number of log events returned
    :type returned: int
    :param total: Total number of log events available for this query
    :type total: float
    """

    _service = "events"
    _action = "get_task_log"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "events": {
                "description": "Log items list",
                "items": {"type": "object"},
                "type": ["array", "null"],
            },
            "returned": {
                "description": "Number of log events returned",
                "type": ["integer", "null"],
            },
            "total": {
                "description": "Total number of log events available for this query",
                "type": ["number", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, events=None, returned=None, total=None, **kwargs):
        super(GetTaskLogResponse, self).__init__(**kwargs)
        self.events = events
        self.returned = returned
        self.total = total

    @schema_property("events")
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

    @schema_property("returned")
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

    @schema_property("total")
    def total(self):
        return self._property_total

    @total.setter
    def total(self, value):
        if value is None:
            self._property_total = None
            return

        self.assert_isinstance(value, "total", six.integer_types + (float,))
        self._property_total = value


class GetTaskMetricsRequest(Request):
    """
    For each task, get a list of metrics for which the requested event type was reported

    :param tasks: Task IDs
    :type tasks: Sequence[str]
    :param event_type: Event type
    :type event_type: EventTypeEnum
    """

    _service = "events"
    _action = "get_task_metrics"
    _version = "2.20"
    _schema = {
        "definitions": {
            "event_type_enum": {
                "enum": [
                    "training_stats_scalar",
                    "training_stats_vector",
                    "training_debug_image",
                    "plot",
                    "log",
                ],
                "type": "string",
            }
        },
        "properties": {
            "event_type": {
                "$ref": "#/definitions/event_type_enum",
                "description": "Event type",
            },
            "tasks": {
                "description": "Task IDs",
                "items": {"type": "string"},
                "type": "array",
            },
        },
        "required": ["tasks"],
        "type": "object",
    }

    def __init__(self, tasks, event_type=None, **kwargs):
        super(GetTaskMetricsRequest, self).__init__(**kwargs)
        self.tasks = tasks
        self.event_type = event_type

    @schema_property("tasks")
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

    @schema_property("event_type")
    def event_type(self):
        return self._property_event_type

    @event_type.setter
    def event_type(self, value):
        if value is None:
            self._property_event_type = None
            return
        if isinstance(value, six.string_types):
            try:
                value = EventTypeEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "event_type", enum.Enum)
        self._property_event_type = value


class GetTaskMetricsResponse(Response):
    """
    Response of events.get_task_metrics endpoint.

    :param metrics: List of task with their metrics
    :type metrics: Sequence[dict]
    """

    _service = "events"
    _action = "get_task_metrics"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "metrics": {
                "description": "List of task with their metrics",
                "items": {"type": "object"},
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, metrics=None, **kwargs):
        super(GetTaskMetricsResponse, self).__init__(**kwargs)
        self.metrics = metrics

    @schema_property("metrics")
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


class GetTaskPlotsRequest(Request):
    """
    Get all 'plot' events for this task

    :param task: Task ID
    :type task: str
    :param iters: Max number of latest iterations for which to return debug images
    :type iters: int
    :param scroll_id: Scroll ID of previous call (used for getting more results)
    :type scroll_id: str
    :param metrics: List of metrics and variants
    :type metrics: Sequence[MetricVariants]
    :param no_scroll: If Truethen no scroll is created. Suitable for one time calls
    :type no_scroll: bool
    """

    _service = "events"
    _action = "get_task_plots"
    _version = "2.20"
    _schema = {
        "definitions": {
            "metric_variants": {
                "metric": {"description": "The metric name", "type": "string"},
                "type": "object",
                "variants": {
                    "description": "The names of the metric variants",
                    "items": {"type": "string"},
                    "type": "array",
                },
            }
        },
        "properties": {
            "iters": {
                "description": "Max number of latest iterations for which to return debug images",
                "type": "integer",
            },
            "metrics": {
                "description": "List of metrics and variants",
                "items": {"$ref": "#/definitions/metric_variants"},
                "type": "array",
            },
            "no_scroll": {
                "default": False,
                "description": "If Truethen no scroll is created. Suitable for one time calls",
                "type": "boolean",
            },
            "scroll_id": {
                "description": "Scroll ID of previous call (used for getting more results)",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(self, task, iters=None, scroll_id=None, metrics=None, no_scroll=False, **kwargs):
        super(GetTaskPlotsRequest, self).__init__(**kwargs)
        self.task = task
        self.iters = iters
        self.scroll_id = scroll_id
        self.metrics = metrics
        self.no_scroll = no_scroll

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

    @schema_property("iters")
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

    @schema_property("metrics")
    def metrics(self):
        return self._property_metrics

    @metrics.setter
    def metrics(self, value):
        if value is None:
            self._property_metrics = None
            return

        self.assert_isinstance(value, "metrics", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [MetricVariants.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "metrics", MetricVariants, is_array=True)
        self._property_metrics = value

    @schema_property("no_scroll")
    def no_scroll(self):
        return self._property_no_scroll

    @no_scroll.setter
    def no_scroll(self, value):
        if value is None:
            self._property_no_scroll = None
            return

        self.assert_isinstance(value, "no_scroll", (bool,))
        self._property_no_scroll = value


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
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "plots": {
                "description": "Plots list",
                "items": {"type": "object"},
                "type": ["array", "null"],
            },
            "returned": {
                "description": "Number of results returned",
                "type": ["integer", "null"],
            },
            "scroll_id": {
                "description": "Scroll ID for getting more results",
                "type": ["string", "null"],
            },
            "total": {
                "description": "Total number of results available for this query",
                "type": ["number", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, plots=None, returned=None, total=None, scroll_id=None, **kwargs):
        super(GetTaskPlotsResponse, self).__init__(**kwargs)
        self.plots = plots
        self.returned = returned
        self.total = total
        self.scroll_id = scroll_id

    @schema_property("plots")
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

    @schema_property("returned")
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

    @schema_property("total")
    def total(self):
        return self._property_total

    @total.setter
    def total(self, value):
        if value is None:
            self._property_total = None
            return

        self.assert_isinstance(value, "total", six.integer_types + (float,))
        self._property_total = value

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


class GetTaskSingleValueMetricsRequest(Request):
    """
    Get single value metrics for the passed tasks

    :param tasks: List of task Task IDs
    :type tasks: Sequence[str]
    """

    _service = "events"
    _action = "get_task_single_value_metrics"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "tasks": {
                "description": "List of task Task IDs",
                "items": {"description": "Task ID", "type": "string"},
                "type": "array",
            }
        },
        "required": ["tasks"],
        "type": "object",
    }

    def __init__(self, tasks, **kwargs):
        super(GetTaskSingleValueMetricsRequest, self).__init__(**kwargs)
        self.tasks = tasks

    @schema_property("tasks")
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


class GetTaskSingleValueMetricsResponse(Response):
    """
    Response of events.get_task_single_value_metrics endpoint.

    :param tasks: Single value metrics grouped by task
    :type tasks: Sequence[dict]
    """

    _service = "events"
    _action = "get_task_single_value_metrics"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "tasks": {
                "description": "Single value metrics grouped by task",
                "items": {
                    "properties": {
                        "task": {"description": "Task ID", "type": "string"},
                        "values": {
                            "items": {
                                "properties": {
                                    "metric": {"type": "string"},
                                    "timestamp": {"type": "number"},
                                    "value": {"type": "number"},
                                    "variant": {"type": "string"},
                                },
                                "type": "object",
                            },
                            "type": "array",
                        },
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, tasks=None, **kwargs):
        super(GetTaskSingleValueMetricsResponse, self).__init__(**kwargs)
        self.tasks = tasks

    @schema_property("tasks")
    def tasks(self):
        return self._property_tasks

    @tasks.setter
    def tasks(self, value):
        if value is None:
            self._property_tasks = None
            return

        self.assert_isinstance(value, "tasks", (list, tuple))

        self.assert_isinstance(value, "tasks", (dict,), is_array=True)
        self._property_tasks = value


class GetVectorMetricsAndVariantsRequest(Request):
    """
    :param task: Task ID
    :type task: str
    """

    _service = "events"
    _action = "get_vector_metrics_and_variants"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {"task": {"description": "Task ID", "type": "string"}},
        "required": ["task"],
        "type": "object",
    }

    def __init__(self, task, **kwargs):
        super(GetVectorMetricsAndVariantsRequest, self).__init__(**kwargs)
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


class GetVectorMetricsAndVariantsResponse(Response):
    """
    Response of events.get_vector_metrics_and_variants endpoint.

    :param metrics:
    :type metrics: Sequence[dict]
    """

    _service = "events"
    _action = "get_vector_metrics_and_variants"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "metrics": {
                "description": "",
                "items": {"type": "object"},
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, metrics=None, **kwargs):
        super(GetVectorMetricsAndVariantsResponse, self).__init__(**kwargs)
        self.metrics = metrics

    @schema_property("metrics")
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

    :param tasks: List of task Task IDs. Maximum amount of tasks is 10
    :type tasks: Sequence[str]
    :param samples: The amount of histogram points to return. Optional, the default
        value is 6000
    :type samples: int
    :param key: Histogram x axis to use: iter - iteration number iso_time - event
        time as ISO formatted string timestamp - event timestamp as milliseconds since
        epoch
    :type key: ScalarKeyEnum
    """

    _service = "events"
    _action = "multi_task_scalar_metrics_iter_histogram"
    _version = "2.20"
    _schema = {
        "definitions": {
            "scalar_key_enum": {
                "enum": ["iter", "timestamp", "iso_time"],
                "type": "string",
            }
        },
        "properties": {
            "key": {
                "$ref": "#/definitions/scalar_key_enum",
                "description": (
                    "\n                        Histogram x axis to use:\n                        iter - iteration"
                    " number\n                        iso_time - event time as ISO formatted string\n                  "
                    "      timestamp - event timestamp as milliseconds since epoch\n                        "
                ),
            },
            "samples": {
                "description": "The amount of histogram points to return. Optional, the default value is 6000",
                "type": "integer",
            },
            "tasks": {
                "description": "List of task Task IDs. Maximum amount of tasks is 10",
                "items": {"description": "List of task Task IDs", "type": "string"},
                "type": "array",
            },
        },
        "required": ["tasks"],
        "type": "object",
    }

    def __init__(self, tasks, samples=None, key=None, **kwargs):
        super(MultiTaskScalarMetricsIterHistogramRequest, self).__init__(**kwargs)
        self.tasks = tasks
        self.samples = samples
        self.key = key

    @schema_property("tasks")
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

    @schema_property("samples")
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

    @schema_property("key")
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
    _version = "2.20"

    _schema = {"additionalProperties": True, "definitions": {}, "type": "object"}


class NextDebugImageSampleRequest(Request):
    """
    Get the image for the next variant for the same iteration or for the next iteration

    :param task: Task ID
    :type task: str
    :param scroll_id: Scroll ID from the previous call to get_debug_image_sample
    :type scroll_id: str
    :param navigate_earlier: If set then get the either previous variant event from
        the current iteration or (if does not exist) the last variant event from the
        previous iteration. Otherwise next variant event from the current iteration or
        first variant event from the next iteration
    :type navigate_earlier: bool
    """

    _service = "events"
    _action = "next_debug_image_sample"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "navigate_earlier": {
                "description": (
                    "If set then get the either previous variant event from the current iteration or (if does not"
                    " exist) the last variant event from the previous iteration. Otherwise next variant event from the"
                    " current iteration or first variant event from the next iteration"
                ),
                "type": "boolean",
            },
            "scroll_id": {
                "description": "Scroll ID from the previous call to get_debug_image_sample",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task", "scroll_id"],
        "type": "object",
    }

    def __init__(self, task, scroll_id, navigate_earlier=None, **kwargs):
        super(NextDebugImageSampleRequest, self).__init__(**kwargs)
        self.task = task
        self.scroll_id = scroll_id
        self.navigate_earlier = navigate_earlier

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

    @schema_property("navigate_earlier")
    def navigate_earlier(self):
        return self._property_navigate_earlier

    @navigate_earlier.setter
    def navigate_earlier(self, value):
        if value is None:
            self._property_navigate_earlier = None
            return

        self.assert_isinstance(value, "navigate_earlier", (bool,))
        self._property_navigate_earlier = value


class NextDebugImageSampleResponse(Response):
    """
    Response of events.next_debug_image_sample endpoint.

    """

    _service = "events"
    _action = "next_debug_image_sample"
    _version = "2.20"

    _schema = {
        "$ref": "#/definitions/debug_image_sample_response",
        "definitions": {
            "debug_image_sample_response": {
                "properties": {
                    "event": {
                        "description": "Debug image event",
                        "type": ["object", "null"],
                    },
                    "max_iteration": {
                        "description": "maximal valid iteration for the variant",
                        "type": ["integer", "null"],
                    },
                    "min_iteration": {
                        "description": "minimal valid iteration for the variant",
                        "type": ["integer", "null"],
                    },
                    "scroll_id": {
                        "description": (
                            "Scroll ID to pass to the next calls to get_debug_image_sample or next_debug_image_sample"
                        ),
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            }
        },
    }


class NextPlotSampleRequest(Request):
    """
    Get the plot for the next variant for the same iteration or for the next iteration

    :param task: Task ID
    :type task: str
    :param scroll_id: Scroll ID from the previous call to get_plot_sample
    :type scroll_id: str
    :param navigate_earlier: If set then get the either previous variant event from
        the current iteration or (if does not exist) the last variant event from the
        previous iteration. Otherwise next variant event from the current iteration or
        first variant event from the next iteration
    :type navigate_earlier: bool
    """

    _service = "events"
    _action = "next_plot_sample"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "navigate_earlier": {
                "description": (
                    "If set then get the either previous variant event from the current "
                    "iteration or (if does not exist) the last variant event from the previous iteration. "
                    "Otherwise next variant event from the current iteration or first variant event from the "
                    "next iteration"
                ),
                "type": "boolean",
            },
            "scroll_id": {
                "description": "Scroll ID from the previous call to get_plot_sample",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task", "scroll_id"],
        "type": "object",
    }

    def __init__(self, task, scroll_id, navigate_earlier=None, **kwargs):
        super(NextPlotSampleRequest, self).__init__(**kwargs)
        self.task = task
        self.scroll_id = scroll_id
        self.navigate_earlier = navigate_earlier

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

    @schema_property("navigate_earlier")
    def navigate_earlier(self):
        return self._property_navigate_earlier

    @navigate_earlier.setter
    def navigate_earlier(self, value):
        if value is None:
            self._property_navigate_earlier = None
            return

        self.assert_isinstance(value, "navigate_earlier", (bool,))
        self._property_navigate_earlier = value


class NextPlotSampleResponse(Response):
    """
    Response of events.next_plot_sample endpoint.

    """

    _service = "events"
    _action = "next_plot_sample"
    _version = "2.20"

    _schema = {
        "$ref": "#/definitions/plot_sample_response",
        "definitions": {
            "plot_sample_response": {
                "properties": {
                    "event": {
                        "description": "Plot event",
                        "type": ["object", "null"],
                    },
                    "max_iteration": {
                        "description": "maximal valid iteration for the variant",
                        "type": ["integer", "null"],
                    },
                    "min_iteration": {
                        "description": "minimal valid iteration for the variant",
                        "type": ["integer", "null"],
                    },
                    "scroll_id": {
                        "description": "Scroll ID to pass to the next calls to get_plot_sample or next_plot_sample",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            }
        },
    }


class PlotsRequest(Request):
    """
    Get plot events for the requested amount of iterations per each task

    :param metrics: List of metrics and variants
    :type metrics: Sequence[TaskMetricVariants]
    :param iters: Max number of latest iterations for which to return debug images
    :type iters: int
    :param navigate_earlier: If set then events are retreived from latest
        iterations to earliest ones. Otherwise from earliest iterations to the latest.
        The default is True
    :type navigate_earlier: bool
    :param refresh: If set then scroll will be moved to the latest iterations. The
        default is False
    :type refresh: bool
    :param scroll_id: Scroll ID of previous call (used for getting more results)
    :type scroll_id: str
    """

    _service = "events"
    _action = "plots"
    _version = "2.20"
    _schema = {
        "definitions": {
            "task_metric_variants": {
                "properties": {
                    "metric": {"description": "Metric name", "type": "string"},
                    "task": {"description": "Task ID", "type": "string"},
                    "variants": {
                        "description": "Metric variant names",
                        "items": {"type": "string"},
                        "type": "array",
                    },
                },
                "required": ["task"],
                "type": "object",
            }
        },
        "properties": {
            "iters": {
                "description": "Max number of latest iterations for which to return debug images",
                "type": "integer",
            },
            "metrics": {
                "description": "List of metrics and variants",
                "items": {"$ref": "#/definitions/task_metric_variants"},
                "type": "array",
            },
            "navigate_earlier": {
                "description": (
                    "If set then events are retreived from latest iterations to earliest ones. "
                    "Otherwise from earliest iterations to the latest. The default is True"
                ),
                "type": "boolean",
            },
            "refresh": {
                "description": "If set then scroll will be moved to the latest iterations. The default is False",
                "type": "boolean",
            },
            "scroll_id": {
                "description": "Scroll ID of previous call (used for getting more results)",
                "type": "string",
            },
        },
        "required": ["metrics"],
        "type": "object",
    }

    def __init__(self, metrics, iters=None, navigate_earlier=None, refresh=None, scroll_id=None, **kwargs):
        super(PlotsRequest, self).__init__(**kwargs)
        self.metrics = metrics
        self.iters = iters
        self.navigate_earlier = navigate_earlier
        self.refresh = refresh
        self.scroll_id = scroll_id

    @schema_property("metrics")
    def metrics(self):
        return self._property_metrics

    @metrics.setter
    def metrics(self, value):
        if value is None:
            self._property_metrics = None
            return

        self.assert_isinstance(value, "metrics", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [TaskMetricVariants.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "metrics", TaskMetricVariants, is_array=True)
        self._property_metrics = value

    @schema_property("iters")
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

    @schema_property("navigate_earlier")
    def navigate_earlier(self):
        return self._property_navigate_earlier

    @navigate_earlier.setter
    def navigate_earlier(self, value):
        if value is None:
            self._property_navigate_earlier = None
            return

        self.assert_isinstance(value, "navigate_earlier", (bool,))
        self._property_navigate_earlier = value

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


class ScalarMetricsIterHistogramRequest(Request):
    """
    Get histogram data of all the vector metrics and variants in the task

    :param task: Task ID
    :type task: str
    :param samples: The amount of histogram points to return (0 to return all the
        points). Optional, the default value is 6000.
    :type samples: int
    :param key: Histogram x axis to use: iter - iteration number iso_time - event
        time as ISO formatted string timestamp - event timestamp as milliseconds since epoch
    :type key: ScalarKeyEnum
    :param metrics: List of metrics and variants
    :type metrics: Sequence[MetricVariants]
    """

    _service = "events"
    _action = "scalar_metrics_iter_histogram"
    _version = "2.20"
    _schema = {
        "definitions": {
            "metric_variants": {
                "metric": {"description": "The metric name", "type": "string"},
                "type": "object",
                "variants": {
                    "description": "The names of the metric variants",
                    "items": {"type": "string"},
                    "type": "array",
                },
            },
            "scalar_key_enum": {
                "enum": ["iter", "timestamp", "iso_time"],
                "type": "string",
            },
        },
        "properties": {
            "key": {
                "$ref": "#/definitions/scalar_key_enum",
                "description": (
                    "Histogram x axis to use:"
                    "iter - iteration number"
                    "iso_time - event time as ISO formatted string"
                    "timestamp - event timestamp as milliseconds since epoch"
                ),
            },
            "metrics": {
                "description": "List of metrics and variants",
                "items": {"$ref": "#/definitions/metric_variants"},
                "type": "array",
            },
            "samples": {
                "description": (
                    "The amount of histogram points to return (0 to return all the points). Optional, the default value"
                    " is 6000."
                ),
                "type": "integer",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task"],
        "type": "object",
    }

    def __init__(self, task, samples=None, key=None, metrics=None, **kwargs):
        super(ScalarMetricsIterHistogramRequest, self).__init__(**kwargs)
        self.task = task
        self.samples = samples
        self.key = key
        self.metrics = metrics

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

    @schema_property("samples")
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

    @schema_property("key")
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

    @schema_property("metrics")
    def metrics(self):
        return self._property_metrics

    @metrics.setter
    def metrics(self, value):
        if value is None:
            self._property_metrics = None
            return

        self.assert_isinstance(value, "metrics", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [MetricVariants.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "metrics", MetricVariants, is_array=True)
        self._property_metrics = value


class ScalarMetricsIterHistogramResponse(Response):
    """
    Response of events.scalar_metrics_iter_histogram endpoint.

    :param images:
    :type images: Sequence[dict]
    """

    _service = "events"
    _action = "scalar_metrics_iter_histogram"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {"images": {"items": {"type": "object"}, "type": ["array", "null"]}},
        "type": "object",
    }

    def __init__(self, images=None, **kwargs):
        super(ScalarMetricsIterHistogramResponse, self).__init__(**kwargs)
        self.images = images

    @schema_property("images")
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


class ScalarMetricsIterRawRequest(Request):
    """
    Get raw data for a specific metric variants in the task

    :param task: Task ID
    :type task: str
    :param metric: Metric and variants for which to return data points
    :type metric: MetricVariants
    :param key: Array of x axis to return. Supported values: iter - iteration
        number timestamp - event timestamp as milliseconds since epoch
    :type key: ScalarKeyEnum
    :param batch_size: The number of data points to return for this call. Optional,
        the default value is 10000. Maximum batch size is 200000
    :type batch_size: int
    :param count_total: Count the total number of data points. If false, total
        number of data points is not counted and null is returned
    :type count_total: bool
    :param scroll_id: Optional Scroll ID. Use to get more data points following a
        previous call
    :type scroll_id: str
    """

    _service = "events"
    _action = "scalar_metrics_iter_raw"
    _version = "2.20"
    _schema = {
        "definitions": {
            "metric_variants": {
                "metric": {"description": "The metric name", "type": "string"},
                "type": "object",
                "variants": {
                    "description": "The names of the metric variants",
                    "items": {"type": "string"},
                    "type": "array",
                },
            },
            "scalar_key_enum": {
                "enum": ["iter", "timestamp", "iso_time"],
                "type": "string",
            },
        },
        "properties": {
            "batch_size": {
                "default": 10000,
                "description": (
                    "The number of data points to return for this call. Optional, the default value is 10000. "
                    " Maximum batch size is 200000"
                ),
                "type": "integer",
            },
            "count_total": {
                "default": False,
                "description": (
                    "Count the total number of data points. If false, total number of data points is not "
                    "counted and null is returned"
                ),
                "type": "boolean",
            },
            "key": {
                "$ref": "#/definitions/scalar_key_enum",
                "description": (
                    "Array of x axis to return. Supported values:"
                    "iter - iteration number"
                    "timestamp - event timestamp as milliseconds since epoch"
                ),
            },
            "metric": {
                "$ref": "#/definitions/metric_variants",
                "description": "Metric and variants for which to return data points",
            },
            "scroll_id": {
                "description": "Optional Scroll ID. Use to get more data points following a previous call",
                "type": "string",
            },
            "task": {"description": "Task ID", "type": "string"},
        },
        "required": ["task", "metric"],
        "type": "object",
    }

    def __init__(self, task, metric, key=None, batch_size=10000, count_total=False, scroll_id=None, **kwargs):
        super(ScalarMetricsIterRawRequest, self).__init__(**kwargs)
        self.task = task
        self.metric = metric
        self.key = key
        self.batch_size = batch_size
        self.count_total = count_total
        self.scroll_id = scroll_id

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

    @schema_property("metric")
    def metric(self):
        return self._property_metric

    @metric.setter
    def metric(self, value):
        if value is None:
            self._property_metric = None
            return
        if isinstance(value, dict):
            value = MetricVariants.from_dict(value)
        else:
            self.assert_isinstance(value, "metric", MetricVariants)
        self._property_metric = value

    @schema_property("key")
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

    @schema_property("batch_size")
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

    @schema_property("count_total")
    def count_total(self):
        return self._property_count_total

    @count_total.setter
    def count_total(self, value):
        if value is None:
            self._property_count_total = None
            return

        self.assert_isinstance(value, "count_total", (bool,))
        self._property_count_total = value

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


class ScalarMetricsIterRawResponse(Response):
    """
    Response of events.scalar_metrics_iter_raw endpoint.

    :param variants: Raw data points for each variant
    :type variants: dict
    :param total: Total data points count. If count_total is false, null is
        returned
    :type total: int
    :param returned: Number of data points returned in this call. If 0 results were
        returned, no more results are avilable
    :type returned: int
    :param scroll_id: Scroll ID. Use to get more data points when calling this
        endpoint again
    :type scroll_id: str
    """

    _service = "events"
    _action = "scalar_metrics_iter_raw"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "returned": {
                "description": (
                    "Number of data points returned in this call. If 0 results were "
                    "returned, no more results are avilable"
                ),
                "type": ["integer", "null"],
            },
            "scroll_id": {
                "description": "Scroll ID. Use to get more data points when calling this endpoint again",
                "type": ["string", "null"],
            },
            "total": {
                "description": "Total data points count. If count_total is false, null is returned",
                "type": ["integer", "null"],
            },
            "variants": {
                "additionalProperties": True,
                "description": "Raw data points for each variant",
                "type": ["object", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, variants=None, total=None, returned=None, scroll_id=None, **kwargs):
        super(ScalarMetricsIterRawResponse, self).__init__(**kwargs)
        self.variants = variants
        self.total = total
        self.returned = returned
        self.scroll_id = scroll_id

    @schema_property("variants")
    def variants(self):
        return self._property_variants

    @variants.setter
    def variants(self, value):
        if value is None:
            self._property_variants = None
            return

        self.assert_isinstance(value, "variants", (dict,))
        self._property_variants = value

    @schema_property("total")
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

    @schema_property("returned")
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
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "metric": {"description": "", "type": "string"},
            "task": {"description": "Task ID", "type": "string"},
            "variant": {"description": "", "type": "string"},
        },
        "required": ["task", "metric", "variant"],
        "type": "object",
    }

    def __init__(self, task, metric, variant, **kwargs):
        super(VectorMetricsIterHistogramRequest, self).__init__(**kwargs)
        self.task = task
        self.metric = metric
        self.variant = variant

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

    @schema_property("metric")
    def metric(self):
        return self._property_metric

    @metric.setter
    def metric(self, value):
        if value is None:
            self._property_metric = None
            return

        self.assert_isinstance(value, "metric", six.string_types)
        self._property_metric = value

    @schema_property("variant")
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
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {"images": {"items": {"type": "object"}, "type": ["array", "null"]}},
        "type": "object",
    }

    def __init__(self, images=None, **kwargs):
        super(VectorMetricsIterHistogramResponse, self).__init__(**kwargs)
        self.images = images

    @schema_property("images")
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
    PlotsRequest: PlotsResponse,
    GetDebugImageSampleRequest: GetDebugImageSampleResponse,
    NextDebugImageSampleRequest: NextDebugImageSampleResponse,
    GetPlotSampleRequest: GetPlotSampleResponse,
    NextPlotSampleRequest: NextPlotSampleResponse,
    GetTaskMetricsRequest: GetTaskMetricsResponse,
    GetTaskLogRequest: GetTaskLogResponse,
    GetTaskEventsRequest: GetTaskEventsResponse,
    DownloadTaskLogRequest: DownloadTaskLogResponse,
    GetTaskPlotsRequest: GetTaskPlotsResponse,
    GetMultiTaskPlotsRequest: GetMultiTaskPlotsResponse,
    GetVectorMetricsAndVariantsRequest: GetVectorMetricsAndVariantsResponse,
    VectorMetricsIterHistogramRequest: VectorMetricsIterHistogramResponse,
    ScalarMetricsIterHistogramRequest: ScalarMetricsIterHistogramResponse,
    MultiTaskScalarMetricsIterHistogramRequest: MultiTaskScalarMetricsIterHistogramResponse,
    GetTaskSingleValueMetricsRequest: GetTaskSingleValueMetricsResponse,
    GetTaskLatestScalarValuesRequest: GetTaskLatestScalarValuesResponse,
    GetScalarMetricsAndVariantsRequest: GetScalarMetricsAndVariantsResponse,
    GetScalarMetricDataRequest: GetScalarMetricDataResponse,
    ScalarMetricsIterRawRequest: ScalarMetricsIterRawResponse,
    ClearScrollRequest: ClearScrollResponse,
    ClearTaskLogRequest: ClearTaskLogResponse,
}
