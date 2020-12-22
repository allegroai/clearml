"""
workers service

Provides an API for worker machines, allowing workers to report status and get tasks for execution
"""
import enum
from datetime import datetime

import six
from dateutil.parser import parse as parse_datetime

from ....backend_api.session import NonStrictDataModel, Request, Response, schema_property, StringEnum


class MetricsCategory(NonStrictDataModel):
    """
    :param name: Name of the metrics category.
    :type name: str
    :param metric_keys: The names of the metrics in the category.
    :type metric_keys: Sequence[str]
    """
    _schema = {
        'properties': {
            'metric_keys': {
                'description': 'The names of the metrics in the category.',
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
            'name': {
                'description': 'Name of the metrics category.',
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, name=None, metric_keys=None, **kwargs):
        super(MetricsCategory, self).__init__(**kwargs)
        self.name = name
        self.metric_keys = metric_keys

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

    @schema_property('metric_keys')
    def metric_keys(self):
        return self._property_metric_keys

    @metric_keys.setter
    def metric_keys(self, value):
        if value is None:
            self._property_metric_keys = None
            return

        self.assert_isinstance(value, "metric_keys", (list, tuple))

        self.assert_isinstance(value, "metric_keys", six.string_types, is_array=True)
        self._property_metric_keys = value


class AggregationType(StringEnum):
    avg = "avg"
    min = "min"
    max = "max"


class StatItem(NonStrictDataModel):
    """
    :param key: Name of a metric
    :type key: str
    :param category:
    :type category: AggregationType
    """
    _schema = {
        'properties': {
            'category': {
                'oneOf': [
                    {'$ref': '#/definitions/aggregation_type'},
                    {'type': 'null'},
                ],
            },
            'key': {'description': 'Name of a metric', 'type': ['string', 'null']},
        },
        'type': 'object',
    }

    def __init__(
            self, key=None, category=None, **kwargs):
        super(StatItem, self).__init__(**kwargs)
        self.key = key
        self.category = category

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

    @schema_property('category')
    def category(self):
        return self._property_category

    @category.setter
    def category(self, value):
        if value is None:
            self._property_category = None
            return
        if isinstance(value, six.string_types):
            try:
                value = AggregationType(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "category", enum.Enum)
        self._property_category = value


class AggregationStats(NonStrictDataModel):
    """
    :param aggregation:
    :type aggregation: AggregationType
    :param values: List of values corresponding to the dates in metric statistics
    :type values: Sequence[float]
    """
    _schema = {
        'properties': {
            'aggregation': {
                'oneOf': [
                    {'$ref': '#/definitions/aggregation_type'},
                    {'type': 'null'},
                ],
            },
            'values': {
                'description': 'List of values corresponding to the dates in metric statistics',
                'items': {'type': 'number'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, aggregation=None, values=None, **kwargs):
        super(AggregationStats, self).__init__(**kwargs)
        self.aggregation = aggregation
        self.values = values

    @schema_property('aggregation')
    def aggregation(self):
        return self._property_aggregation

    @aggregation.setter
    def aggregation(self, value):
        if value is None:
            self._property_aggregation = None
            return
        if isinstance(value, six.string_types):
            try:
                value = AggregationType(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "aggregation", enum.Enum)
        self._property_aggregation = value

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


class MetricStats(NonStrictDataModel):
    """
    :param metric: Name of the metric (cpu_usage, memory_used etc.)
    :type metric: str
    :param variant: Name of the metric component. Set only if 'split_by_variant'
        was set in the request
    :type variant: str
    :param dates: List of timestamps (in seconds from epoch) in the acceding order.
        The timestamps are separated by the requested interval. Timestamps where no
        workers activity was recorded are omitted.
    :type dates: Sequence[int]
    :param stats: Statistics data by type
    :type stats: Sequence[AggregationStats]
    """
    _schema = {
        'properties': {
            'dates': {
                'description': 'List of timestamps (in seconds from epoch) in the acceding order. The timestamps are separated by the requested interval. Timestamps where no workers activity was recorded are omitted.',
                'items': {'type': 'integer'},
                'type': ['array', 'null'],
            },
            'metric': {
                'description': 'Name of the metric (cpu_usage, memory_used etc.)',
                'type': ['string', 'null'],
            },
            'stats': {
                'description': 'Statistics data by type',
                'items': {'$ref': '#/definitions/aggregation_stats'},
                'type': ['array', 'null'],
            },
            'variant': {
                'description': "Name of the metric component. Set only if 'split_by_variant' was set in the request",
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, metric=None, variant=None, dates=None, stats=None, **kwargs):
        super(MetricStats, self).__init__(**kwargs)
        self.metric = metric
        self.variant = variant
        self.dates = dates
        self.stats = stats

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

    @schema_property('dates')
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

    @schema_property('stats')
    def stats(self):
        return self._property_stats

    @stats.setter
    def stats(self, value):
        if value is None:
            self._property_stats = None
            return

        self.assert_isinstance(value, "stats", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [AggregationStats.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "stats", AggregationStats, is_array=True)
        self._property_stats = value


class WorkerStats(NonStrictDataModel):
    """
    :param worker: ID of the worker
    :type worker: str
    :param metrics: List of the metrics statistics for the worker
    :type metrics: Sequence[MetricStats]
    """
    _schema = {
        'properties': {
            'metrics': {
                'description': 'List of the metrics statistics for the worker',
                'items': {'$ref': '#/definitions/metric_stats'},
                'type': ['array', 'null'],
            },
            'worker': {
                'description': 'ID of the worker',
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, worker=None, metrics=None, **kwargs):
        super(WorkerStats, self).__init__(**kwargs)
        self.worker = worker
        self.metrics = metrics

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

    @schema_property('metrics')
    def metrics(self):
        return self._property_metrics

    @metrics.setter
    def metrics(self, value):
        if value is None:
            self._property_metrics = None
            return

        self.assert_isinstance(value, "metrics", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [MetricStats.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "metrics", MetricStats, is_array=True)
        self._property_metrics = value


class ActivitySeries(NonStrictDataModel):
    """
    :param dates: List of timestamps (in seconds from epoch) in the acceding order.
        The timestamps are separated by the requested interval.
    :type dates: Sequence[int]
    :param counts: List of worker counts corresponding to the timestamps in the
        dates list. None values are returned for the dates with no workers.
    :type counts: Sequence[int]
    """
    _schema = {
        'properties': {
            'counts': {
                'description': 'List of worker counts corresponding to the timestamps in the dates list. None values are returned for the dates with no workers.',
                'items': {'type': 'integer'},
                'type': ['array', 'null'],
            },
            'dates': {
                'description': 'List of timestamps (in seconds from epoch) in the acceding order. The timestamps are separated by the requested interval.',
                'items': {'type': 'integer'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, dates=None, counts=None, **kwargs):
        super(ActivitySeries, self).__init__(**kwargs)
        self.dates = dates
        self.counts = counts

    @schema_property('dates')
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

    @schema_property('counts')
    def counts(self):
        return self._property_counts

    @counts.setter
    def counts(self, value):
        if value is None:
            self._property_counts = None
            return

        self.assert_isinstance(value, "counts", (list, tuple))
        value = [int(v) if isinstance(v, float) and v.is_integer() else v for v in value]

        self.assert_isinstance(value, "counts", six.integer_types, is_array=True)
        self._property_counts = value


class Worker(NonStrictDataModel):
    """
    :param id: Worker ID
    :type id: str
    :param user: Associated user (under whose credentials are used by the worker
        daemon)
    :type user: IdNameEntry
    :param company: Associated company
    :type company: IdNameEntry
    :param ip: IP of the worker
    :type ip: str
    :param register_time: Registration time
    :type register_time: datetime.datetime
    :param last_activity_time: Last activity time (even if an error occurred)
    :type last_activity_time: datetime.datetime
    :param last_report_time: Last successful report time
    :type last_report_time: datetime.datetime
    :param task: Task currently being run by the worker
    :type task: CurrentTaskEntry
    :param queue: Queue from which running task was taken
    :type queue: QueueEntry
    :param queues: List of queues on which the worker is listening
    :type queues: Sequence[QueueEntry]
    """
    _schema = {
        'properties': {
            'company': {
                'description': 'Associated company',
                'oneOf': [
                    {'$ref': '#/definitions/id_name_entry'},
                    {'type': 'null'},
                ],
            },
            'id': {'description': 'Worker ID', 'type': ['string', 'null']},
            'ip': {'description': 'IP of the worker', 'type': ['string', 'null']},
            'last_activity_time': {
                'description': 'Last activity time (even if an error occurred)',
                'format': 'date-time',
                'type': ['string', 'null'],
            },
            'last_report_time': {
                'description': 'Last successful report time',
                'format': 'date-time',
                'type': ['string', 'null'],
            },
            'queue': {
                'description': 'Queue from which running task was taken',
                'oneOf': [{'$ref': '#/definitions/queue_entry'}, {'type': 'null'}],
            },
            'queues': {
                'description': 'List of queues on which the worker is listening',
                'items': {'$ref': '#/definitions/queue_entry'},
                'type': ['array', 'null'],
            },
            'register_time': {
                'description': 'Registration time',
                'format': 'date-time',
                'type': ['string', 'null'],
            },
            'task': {
                'description': 'Task currently being run by the worker',
                'oneOf': [
                    {'$ref': '#/definitions/current_task_entry'},
                    {'type': 'null'},
                ],
            },
            'user': {
                'description': 'Associated user (under whose credentials are used by the worker daemon)',
                'oneOf': [
                    {'$ref': '#/definitions/id_name_entry'},
                    {'type': 'null'},
                ],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, id=None, user=None, company=None, ip=None, register_time=None, last_activity_time=None, last_report_time=None, task=None, queue=None, queues=None, **kwargs):
        super(Worker, self).__init__(**kwargs)
        self.id = id
        self.user = user
        self.company = company
        self.ip = ip
        self.register_time = register_time
        self.last_activity_time = last_activity_time
        self.last_report_time = last_report_time
        self.task = task
        self.queue = queue
        self.queues = queues

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

    @schema_property('user')
    def user(self):
        return self._property_user

    @user.setter
    def user(self, value):
        if value is None:
            self._property_user = None
            return
        if isinstance(value, dict):
            value = IdNameEntry.from_dict(value)
        else:
            self.assert_isinstance(value, "user", IdNameEntry)
        self._property_user = value

    @schema_property('company')
    def company(self):
        return self._property_company

    @company.setter
    def company(self, value):
        if value is None:
            self._property_company = None
            return
        if isinstance(value, dict):
            value = IdNameEntry.from_dict(value)
        else:
            self.assert_isinstance(value, "company", IdNameEntry)
        self._property_company = value

    @schema_property('ip')
    def ip(self):
        return self._property_ip

    @ip.setter
    def ip(self, value):
        if value is None:
            self._property_ip = None
            return

        self.assert_isinstance(value, "ip", six.string_types)
        self._property_ip = value

    @schema_property('register_time')
    def register_time(self):
        return self._property_register_time

    @register_time.setter
    def register_time(self, value):
        if value is None:
            self._property_register_time = None
            return

        self.assert_isinstance(value, "register_time", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_register_time = value

    @schema_property('last_activity_time')
    def last_activity_time(self):
        return self._property_last_activity_time

    @last_activity_time.setter
    def last_activity_time(self, value):
        if value is None:
            self._property_last_activity_time = None
            return

        self.assert_isinstance(value, "last_activity_time", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_last_activity_time = value

    @schema_property('last_report_time')
    def last_report_time(self):
        return self._property_last_report_time

    @last_report_time.setter
    def last_report_time(self, value):
        if value is None:
            self._property_last_report_time = None
            return

        self.assert_isinstance(value, "last_report_time", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_last_report_time = value

    @schema_property('task')
    def task(self):
        return self._property_task

    @task.setter
    def task(self, value):
        if value is None:
            self._property_task = None
            return
        if isinstance(value, dict):
            value = CurrentTaskEntry.from_dict(value)
        else:
            self.assert_isinstance(value, "task", CurrentTaskEntry)
        self._property_task = value

    @schema_property('queue')
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return
        if isinstance(value, dict):
            value = QueueEntry.from_dict(value)
        else:
            self.assert_isinstance(value, "queue", QueueEntry)
        self._property_queue = value

    @schema_property('queues')
    def queues(self):
        return self._property_queues

    @queues.setter
    def queues(self, value):
        if value is None:
            self._property_queues = None
            return

        self.assert_isinstance(value, "queues", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [QueueEntry.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "queues", QueueEntry, is_array=True)
        self._property_queues = value


class IdNameEntry(NonStrictDataModel):
    """
    :param id: Worker ID
    :type id: str
    :param name: Worker name
    :type name: str
    """
    _schema = {
        'properties': {
            'id': {'description': 'Worker ID', 'type': ['string', 'null']},
            'name': {'description': 'Worker name', 'type': ['string', 'null']},
        },
        'type': 'object',
    }

    def __init__(
            self, id=None, name=None, **kwargs):
        super(IdNameEntry, self).__init__(**kwargs)
        self.id = id
        self.name = name

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


class CurrentTaskEntry(NonStrictDataModel):
    """
    :param id: Worker ID
    :type id: str
    :param name: Worker name
    :type name: str
    :param running_time: Task running time
    :type running_time: int
    :param last_iteration: Last task iteration
    :type last_iteration: int
    """
    _schema = {
        'properties': {
            'id': {'description': 'Worker ID', 'type': ['string', 'null']},
            'last_iteration': {
                'description': 'Last task iteration',
                'type': ['integer', 'null'],
            },
            'name': {'description': 'Worker name', 'type': ['string', 'null']},
            'running_time': {
                'description': 'Task running time',
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, id=None, name=None, running_time=None, last_iteration=None, **kwargs):
        super(CurrentTaskEntry, self).__init__(**kwargs)
        self.id = id
        self.name = name
        self.running_time = running_time
        self.last_iteration = last_iteration

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

    @schema_property('running_time')
    def running_time(self):
        return self._property_running_time

    @running_time.setter
    def running_time(self, value):
        if value is None:
            self._property_running_time = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "running_time", six.integer_types)
        self._property_running_time = value

    @schema_property('last_iteration')
    def last_iteration(self):
        return self._property_last_iteration

    @last_iteration.setter
    def last_iteration(self, value):
        if value is None:
            self._property_last_iteration = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "last_iteration", six.integer_types)
        self._property_last_iteration = value


class QueueEntry(NonStrictDataModel):
    """
    :param id: Worker ID
    :type id: str
    :param name: Worker name
    :type name: str
    :param next_task: Next task in the queue
    :type next_task: IdNameEntry
    :param num_tasks: Number of task entries in the queue
    :type num_tasks: int
    """
    _schema = {
        'properties': {
            'id': {'description': 'Worker ID', 'type': ['string', 'null']},
            'name': {'description': 'Worker name', 'type': ['string', 'null']},
            'next_task': {
                'description': 'Next task in the queue',
                'oneOf': [
                    {'$ref': '#/definitions/id_name_entry'},
                    {'type': 'null'},
                ],
            },
            'num_tasks': {
                'description': 'Number of task entries in the queue',
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, id=None, name=None, next_task=None, num_tasks=None, **kwargs):
        super(QueueEntry, self).__init__(**kwargs)
        self.id = id
        self.name = name
        self.next_task = next_task
        self.num_tasks = num_tasks

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

    @schema_property('next_task')
    def next_task(self):
        return self._property_next_task

    @next_task.setter
    def next_task(self, value):
        if value is None:
            self._property_next_task = None
            return
        if isinstance(value, dict):
            value = IdNameEntry.from_dict(value)
        else:
            self.assert_isinstance(value, "next_task", IdNameEntry)
        self._property_next_task = value

    @schema_property('num_tasks')
    def num_tasks(self):
        return self._property_num_tasks

    @num_tasks.setter
    def num_tasks(self, value):
        if value is None:
            self._property_num_tasks = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "num_tasks", six.integer_types)
        self._property_num_tasks = value


class MachineStats(NonStrictDataModel):
    """
    :param cpu_usage: Average CPU usage per core
    :type cpu_usage: Sequence[float]
    :param gpu_usage: Average GPU usage per GPU card
    :type gpu_usage: Sequence[float]
    :param memory_used: Used memory MBs
    :type memory_used: int
    :param memory_free: Free memory MBs
    :type memory_free: int
    :param gpu_memory_free: GPU free memory MBs
    :type gpu_memory_free: Sequence[int]
    :param gpu_memory_used: GPU used memory MBs
    :type gpu_memory_used: Sequence[int]
    :param network_tx: Mbytes per second
    :type network_tx: int
    :param network_rx: Mbytes per second
    :type network_rx: int
    :param disk_free_home: Mbytes free space of /home drive
    :type disk_free_home: int
    :param disk_free_temp: Mbytes free space of /tmp drive
    :type disk_free_temp: int
    :param disk_read: Mbytes read per second
    :type disk_read: int
    :param disk_write: Mbytes write per second
    :type disk_write: int
    :param cpu_temperature: CPU temperature
    :type cpu_temperature: Sequence[float]
    :param gpu_temperature: GPU temperature
    :type gpu_temperature: Sequence[float]
    """
    _schema = {
        'properties': {
            'cpu_temperature': {
                'description': 'CPU temperature',
                'items': {'type': 'number'},
                'type': ['array', 'null'],
            },
            'cpu_usage': {
                'description': 'Average CPU usage per core',
                'items': {'type': 'number'},
                'type': ['array', 'null'],
            },
            'disk_free_home': {
                'description': 'Mbytes free space of /home drive',
                'type': ['integer', 'null'],
            },
            'disk_free_temp': {
                'description': 'Mbytes free space of /tmp drive',
                'type': ['integer', 'null'],
            },
            'disk_read': {
                'description': 'Mbytes read per second',
                'type': ['integer', 'null'],
            },
            'disk_write': {
                'description': 'Mbytes write per second',
                'type': ['integer', 'null'],
            },
            'gpu_memory_free': {
                'description': 'GPU free memory MBs',
                'items': {'type': 'integer'},
                'type': ['array', 'null'],
            },
            'gpu_memory_used': {
                'description': 'GPU used memory MBs',
                'items': {'type': 'integer'},
                'type': ['array', 'null'],
            },
            'gpu_temperature': {
                'description': 'GPU temperature',
                'items': {'type': 'number'},
                'type': ['array', 'null'],
            },
            'gpu_usage': {
                'description': 'Average GPU usage per GPU card',
                'items': {'type': 'number'},
                'type': ['array', 'null'],
            },
            'memory_free': {
                'description': 'Free memory MBs',
                'type': ['integer', 'null'],
            },
            'memory_used': {
                'description': 'Used memory MBs',
                'type': ['integer', 'null'],
            },
            'network_rx': {
                'description': 'Mbytes per second',
                'type': ['integer', 'null'],
            },
            'network_tx': {
                'description': 'Mbytes per second',
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, cpu_usage=None, gpu_usage=None, memory_used=None, memory_free=None, gpu_memory_free=None, gpu_memory_used=None, network_tx=None, network_rx=None, disk_free_home=None, disk_free_temp=None, disk_read=None, disk_write=None, cpu_temperature=None, gpu_temperature=None, **kwargs):
        super(MachineStats, self).__init__(**kwargs)
        self.cpu_usage = cpu_usage
        self.gpu_usage = gpu_usage
        self.memory_used = memory_used
        self.memory_free = memory_free
        self.gpu_memory_free = gpu_memory_free
        self.gpu_memory_used = gpu_memory_used
        self.network_tx = network_tx
        self.network_rx = network_rx
        self.disk_free_home = disk_free_home
        self.disk_free_temp = disk_free_temp
        self.disk_read = disk_read
        self.disk_write = disk_write
        self.cpu_temperature = cpu_temperature
        self.gpu_temperature = gpu_temperature

    @schema_property('cpu_usage')
    def cpu_usage(self):
        return self._property_cpu_usage

    @cpu_usage.setter
    def cpu_usage(self, value):
        if value is None:
            self._property_cpu_usage = None
            return

        self.assert_isinstance(value, "cpu_usage", (list, tuple))

        self.assert_isinstance(value, "cpu_usage", six.integer_types + (float,), is_array=True)
        self._property_cpu_usage = value

    @schema_property('gpu_usage')
    def gpu_usage(self):
        return self._property_gpu_usage

    @gpu_usage.setter
    def gpu_usage(self, value):
        if value is None:
            self._property_gpu_usage = None
            return

        self.assert_isinstance(value, "gpu_usage", (list, tuple))

        self.assert_isinstance(value, "gpu_usage", six.integer_types + (float,), is_array=True)
        self._property_gpu_usage = value

    @schema_property('memory_used')
    def memory_used(self):
        return self._property_memory_used

    @memory_used.setter
    def memory_used(self, value):
        if value is None:
            self._property_memory_used = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "memory_used", six.integer_types)
        self._property_memory_used = value

    @schema_property('memory_free')
    def memory_free(self):
        return self._property_memory_free

    @memory_free.setter
    def memory_free(self, value):
        if value is None:
            self._property_memory_free = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "memory_free", six.integer_types)
        self._property_memory_free = value

    @schema_property('gpu_memory_free')
    def gpu_memory_free(self):
        return self._property_gpu_memory_free

    @gpu_memory_free.setter
    def gpu_memory_free(self, value):
        if value is None:
            self._property_gpu_memory_free = None
            return

        self.assert_isinstance(value, "gpu_memory_free", (list, tuple))
        value = [int(v) if isinstance(v, float) and v.is_integer() else v for v in value]

        self.assert_isinstance(value, "gpu_memory_free", six.integer_types, is_array=True)
        self._property_gpu_memory_free = value

    @schema_property('gpu_memory_used')
    def gpu_memory_used(self):
        return self._property_gpu_memory_used

    @gpu_memory_used.setter
    def gpu_memory_used(self, value):
        if value is None:
            self._property_gpu_memory_used = None
            return

        self.assert_isinstance(value, "gpu_memory_used", (list, tuple))
        value = [int(v) if isinstance(v, float) and v.is_integer() else v for v in value]

        self.assert_isinstance(value, "gpu_memory_used", six.integer_types, is_array=True)
        self._property_gpu_memory_used = value

    @schema_property('network_tx')
    def network_tx(self):
        return self._property_network_tx

    @network_tx.setter
    def network_tx(self, value):
        if value is None:
            self._property_network_tx = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "network_tx", six.integer_types)
        self._property_network_tx = value

    @schema_property('network_rx')
    def network_rx(self):
        return self._property_network_rx

    @network_rx.setter
    def network_rx(self, value):
        if value is None:
            self._property_network_rx = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "network_rx", six.integer_types)
        self._property_network_rx = value

    @schema_property('disk_free_home')
    def disk_free_home(self):
        return self._property_disk_free_home

    @disk_free_home.setter
    def disk_free_home(self, value):
        if value is None:
            self._property_disk_free_home = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "disk_free_home", six.integer_types)
        self._property_disk_free_home = value

    @schema_property('disk_free_temp')
    def disk_free_temp(self):
        return self._property_disk_free_temp

    @disk_free_temp.setter
    def disk_free_temp(self, value):
        if value is None:
            self._property_disk_free_temp = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "disk_free_temp", six.integer_types)
        self._property_disk_free_temp = value

    @schema_property('disk_read')
    def disk_read(self):
        return self._property_disk_read

    @disk_read.setter
    def disk_read(self, value):
        if value is None:
            self._property_disk_read = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "disk_read", six.integer_types)
        self._property_disk_read = value

    @schema_property('disk_write')
    def disk_write(self):
        return self._property_disk_write

    @disk_write.setter
    def disk_write(self, value):
        if value is None:
            self._property_disk_write = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "disk_write", six.integer_types)
        self._property_disk_write = value

    @schema_property('cpu_temperature')
    def cpu_temperature(self):
        return self._property_cpu_temperature

    @cpu_temperature.setter
    def cpu_temperature(self, value):
        if value is None:
            self._property_cpu_temperature = None
            return

        self.assert_isinstance(value, "cpu_temperature", (list, tuple))

        self.assert_isinstance(value, "cpu_temperature", six.integer_types + (float,), is_array=True)
        self._property_cpu_temperature = value

    @schema_property('gpu_temperature')
    def gpu_temperature(self):
        return self._property_gpu_temperature

    @gpu_temperature.setter
    def gpu_temperature(self, value):
        if value is None:
            self._property_gpu_temperature = None
            return

        self.assert_isinstance(value, "gpu_temperature", (list, tuple))

        self.assert_isinstance(value, "gpu_temperature", six.integer_types + (float,), is_array=True)
        self._property_gpu_temperature = value


class GetActivityReportRequest(Request):
    """
    Returns count of active company workers in the selected time range.

    :param from_date: Starting time (in seconds from epoch) for collecting
        statistics
    :type from_date: float
    :param to_date: Ending time (in seconds from epoch) for collecting statistics
    :type to_date: float
    :param interval: Time interval in seconds for a single statistics point. The
        minimal value is 1
    :type interval: int
    """

    _service = "workers"
    _action = "get_activity_report"
    _version = "2.4"
    _schema = {
        'definitions': {},
        'properties': {
            'from_date': {
                'description': 'Starting time (in seconds from epoch) for collecting statistics',
                'type': 'number',
            },
            'interval': {
                'description': 'Time interval in seconds for a single statistics point. The minimal value is 1',
                'type': 'integer',
            },
            'to_date': {
                'description': 'Ending time (in seconds from epoch) for collecting statistics',
                'type': 'number',
            },
        },
        'required': ['from_date', 'to_date', 'interval'],
        'type': 'object',
    }

    def __init__(
            self, from_date, to_date, interval, **kwargs):
        super(GetActivityReportRequest, self).__init__(**kwargs)
        self.from_date = from_date
        self.to_date = to_date
        self.interval = interval

    @schema_property('from_date')
    def from_date(self):
        return self._property_from_date

    @from_date.setter
    def from_date(self, value):
        if value is None:
            self._property_from_date = None
            return

        self.assert_isinstance(value, "from_date", six.integer_types + (float,))
        self._property_from_date = value

    @schema_property('to_date')
    def to_date(self):
        return self._property_to_date

    @to_date.setter
    def to_date(self, value):
        if value is None:
            self._property_to_date = None
            return

        self.assert_isinstance(value, "to_date", six.integer_types + (float,))
        self._property_to_date = value

    @schema_property('interval')
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


class GetActivityReportResponse(Response):
    """
    Response of workers.get_activity_report endpoint.

    :param total: Activity series that include all the workers that sent reports in
        the given time interval.
    :type total: ActivitySeries
    :param active: Activity series that include only workers that worked on a task
        in the given time interval.
    :type active: ActivitySeries
    """
    _service = "workers"
    _action = "get_activity_report"
    _version = "2.4"

    _schema = {
        'definitions': {
            'activity_series': {
                'properties': {
                    'counts': {
                        'description': 'List of worker counts corresponding to the timestamps in the dates list. None values are returned for the dates with no workers.',
                        'items': {'type': 'integer'},
                        'type': ['array', 'null'],
                    },
                    'dates': {
                        'description': 'List of timestamps (in seconds from epoch) in the acceding order. The timestamps are separated by the requested interval.',
                        'items': {'type': 'integer'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
        },
        'properties': {
            'active': {
                'description': 'Activity series that include only workers that worked on a task in the given time interval.',
                'oneOf': [
                    {'$ref': '#/definitions/activity_series'},
                    {'type': 'null'},
                ],
            },
            'total': {
                'description': 'Activity series that include all the workers that sent reports in the given time interval.',
                'oneOf': [
                    {'$ref': '#/definitions/activity_series'},
                    {'type': 'null'},
                ],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, total=None, active=None, **kwargs):
        super(GetActivityReportResponse, self).__init__(**kwargs)
        self.total = total
        self.active = active

    @schema_property('total')
    def total(self):
        return self._property_total

    @total.setter
    def total(self, value):
        if value is None:
            self._property_total = None
            return
        if isinstance(value, dict):
            value = ActivitySeries.from_dict(value)
        else:
            self.assert_isinstance(value, "total", ActivitySeries)
        self._property_total = value

    @schema_property('active')
    def active(self):
        return self._property_active

    @active.setter
    def active(self, value):
        if value is None:
            self._property_active = None
            return
        if isinstance(value, dict):
            value = ActivitySeries.from_dict(value)
        else:
            self.assert_isinstance(value, "active", ActivitySeries)
        self._property_active = value


class GetAllRequest(Request):
    """
    Returns information on all registered workers.

    :param last_seen: Filter out workers not active for more than last_seen
        seconds. A value or 0 or 'none' will disable the filter.
    :type last_seen: int
    """

    _service = "workers"
    _action = "get_all"
    _version = "2.4"
    _schema = {
        'definitions': {},
        'properties': {
            'last_seen': {
                'default': 3600,
                'description': "Filter out workers not active for more than last_seen seconds.\n                            A value or 0 or 'none' will disable the filter.",
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, last_seen=3600, **kwargs):
        super(GetAllRequest, self).__init__(**kwargs)
        self.last_seen = last_seen

    @schema_property('last_seen')
    def last_seen(self):
        return self._property_last_seen

    @last_seen.setter
    def last_seen(self, value):
        if value is None:
            self._property_last_seen = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "last_seen", six.integer_types)
        self._property_last_seen = value


class GetAllResponse(Response):
    """
    Response of workers.get_all endpoint.

    :param workers:
    :type workers: Sequence[Worker]
    """
    _service = "workers"
    _action = "get_all"
    _version = "2.4"

    _schema = {
        'definitions': {
            'current_task_entry': {
                'properties': {
                    'id': {'description': 'Worker ID', 'type': ['string', 'null']},
                    'last_iteration': {
                        'description': 'Last task iteration',
                        'type': ['integer', 'null'],
                    },
                    'name': {
                        'description': 'Worker name',
                        'type': ['string', 'null'],
                    },
                    'running_time': {
                        'description': 'Task running time',
                        'type': ['integer', 'null'],
                    },
                },
                'type': 'object',
            },
            'id_name_entry': {
                'properties': {
                    'id': {'description': 'Worker ID', 'type': ['string', 'null']},
                    'name': {
                        'description': 'Worker name',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'queue_entry': {
                'properties': {
                    'id': {'description': 'Worker ID', 'type': ['string', 'null']},
                    'name': {
                        'description': 'Worker name',
                        'type': ['string', 'null'],
                    },
                    'next_task': {
                        'description': 'Next task in the queue',
                        'oneOf': [
                            {'$ref': '#/definitions/id_name_entry'},
                            {'type': 'null'},
                        ],
                    },
                    'num_tasks': {
                        'description': 'Number of task entries in the queue',
                        'type': ['integer', 'null'],
                    },
                },
                'type': 'object',
            },
            'worker': {
                'properties': {
                    'company': {
                        'description': 'Associated company',
                        'oneOf': [
                            {'$ref': '#/definitions/id_name_entry'},
                            {'type': 'null'},
                        ],
                    },
                    'id': {'description': 'Worker ID', 'type': ['string', 'null']},
                    'ip': {
                        'description': 'IP of the worker',
                        'type': ['string', 'null'],
                    },
                    'last_activity_time': {
                        'description': 'Last activity time (even if an error occurred)',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'last_report_time': {
                        'description': 'Last successful report time',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'queue': {
                        'description': 'Queue from which running task was taken',
                        'oneOf': [
                            {'$ref': '#/definitions/queue_entry'},
                            {'type': 'null'},
                        ],
                    },
                    'queues': {
                        'description': 'List of queues on which the worker is listening',
                        'items': {'$ref': '#/definitions/queue_entry'},
                        'type': ['array', 'null'],
                    },
                    'register_time': {
                        'description': 'Registration time',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'task': {
                        'description': 'Task currently being run by the worker',
                        'oneOf': [
                            {'$ref': '#/definitions/current_task_entry'},
                            {'type': 'null'},
                        ],
                    },
                    'user': {
                        'description': 'Associated user (under whose credentials are used by the worker daemon)',
                        'oneOf': [
                            {'$ref': '#/definitions/id_name_entry'},
                            {'type': 'null'},
                        ],
                    },
                },
                'type': 'object',
            },
        },
        'properties': {
            'workers': {
                'items': {'$ref': '#/definitions/worker'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, workers=None, **kwargs):
        super(GetAllResponse, self).__init__(**kwargs)
        self.workers = workers

    @schema_property('workers')
    def workers(self):
        return self._property_workers

    @workers.setter
    def workers(self, value):
        if value is None:
            self._property_workers = None
            return

        self.assert_isinstance(value, "workers", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Worker.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "workers", Worker, is_array=True)
        self._property_workers = value


class GetMetricKeysRequest(Request):
    """
    Returns worker statistics metric keys grouped by categories.

    :param worker_ids: List of worker ids to collect metrics for. If not provided
        or empty then all the company workers metrics are analyzed.
    :type worker_ids: Sequence[str]
    """

    _service = "workers"
    _action = "get_metric_keys"
    _version = "2.4"
    _schema = {
        'definitions': {},
        'properties': {
            'worker_ids': {
                'description': 'List of worker ids to collect metrics for. If not provided or empty then all the company workers metrics are analyzed.',
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, worker_ids=None, **kwargs):
        super(GetMetricKeysRequest, self).__init__(**kwargs)
        self.worker_ids = worker_ids

    @schema_property('worker_ids')
    def worker_ids(self):
        return self._property_worker_ids

    @worker_ids.setter
    def worker_ids(self, value):
        if value is None:
            self._property_worker_ids = None
            return

        self.assert_isinstance(value, "worker_ids", (list, tuple))

        self.assert_isinstance(value, "worker_ids", six.string_types, is_array=True)
        self._property_worker_ids = value


class GetMetricKeysResponse(Response):
    """
    Response of workers.get_metric_keys endpoint.

    :param categories: List of unique metric categories found in the statistics of
        the requested workers.
    :type categories: Sequence[MetricsCategory]
    """
    _service = "workers"
    _action = "get_metric_keys"
    _version = "2.4"

    _schema = {
        'definitions': {
            'metrics_category': {
                'properties': {
                    'metric_keys': {
                        'description': 'The names of the metrics in the category.',
                        'items': {'type': 'string'},
                        'type': ['array', 'null'],
                    },
                    'name': {
                        'description': 'Name of the metrics category.',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
        },
        'properties': {
            'categories': {
                'description': 'List of unique metric categories found in the statistics of the requested workers.',
                'items': {'$ref': '#/definitions/metrics_category'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, categories=None, **kwargs):
        super(GetMetricKeysResponse, self).__init__(**kwargs)
        self.categories = categories

    @schema_property('categories')
    def categories(self):
        return self._property_categories

    @categories.setter
    def categories(self, value):
        if value is None:
            self._property_categories = None
            return

        self.assert_isinstance(value, "categories", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [MetricsCategory.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "categories", MetricsCategory, is_array=True)
        self._property_categories = value


class GetStatsRequest(Request):
    """
    Returns statistics for the selected workers and time range aggregated by date intervals.

    :param worker_ids: List of worker ids to collect metrics for. If not provided
        or empty then all the company workers metrics are analyzed.
    :type worker_ids: Sequence[str]
    :param from_date: Starting time (in seconds from epoch) for collecting
        statistics
    :type from_date: float
    :param to_date: Ending time (in seconds from epoch) for collecting statistics
    :type to_date: float
    :param interval: Time interval in seconds for a single statistics point. The
        minimal value is 1
    :type interval: int
    :param items: List of metric keys and requested statistics
    :type items: Sequence[StatItem]
    :param split_by_variant: If true then break statistics by hardware sub types
    :type split_by_variant: bool
    """

    _service = "workers"
    _action = "get_stats"
    _version = "2.4"
    _schema = {
        'definitions': {
            'aggregation_type': {
                'description': 'Metric aggregation type',
                'enum': ['avg', 'min', 'max'],
                'type': 'string',
            },
            'stat_item': {
                'properties': {
                    'category': {
                        'oneOf': [
                            {'$ref': '#/definitions/aggregation_type'},
                            {'type': 'null'},
                        ],
                    },
                    'key': {
                        'description': 'Name of a metric',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
        },
        'properties': {
            'from_date': {
                'description': 'Starting time (in seconds from epoch) for collecting statistics',
                'type': 'number',
            },
            'interval': {
                'description': 'Time interval in seconds for a single statistics point. The minimal value is 1',
                'type': 'integer',
            },
            'items': {
                'description': 'List of metric keys and requested statistics',
                'items': {'$ref': '#/definitions/stat_item'},
                'type': 'array',
            },
            'split_by_variant': {
                'default': False,
                'description': 'If true then break statistics by hardware sub types',
                'type': 'boolean',
            },
            'to_date': {
                'description': 'Ending time (in seconds from epoch) for collecting statistics',
                'type': 'number',
            },
            'worker_ids': {
                'description': 'List of worker ids to collect metrics for. If not provided or empty then all the company workers metrics are analyzed.',
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
        },
        'required': ['from_date', 'to_date', 'interval', 'items'],
        'type': 'object',
    }

    def __init__(
            self, from_date, to_date, interval, items, worker_ids=None, split_by_variant=False, **kwargs):
        super(GetStatsRequest, self).__init__(**kwargs)
        self.worker_ids = worker_ids
        self.from_date = from_date
        self.to_date = to_date
        self.interval = interval
        self.items = items
        self.split_by_variant = split_by_variant

    @schema_property('worker_ids')
    def worker_ids(self):
        return self._property_worker_ids

    @worker_ids.setter
    def worker_ids(self, value):
        if value is None:
            self._property_worker_ids = None
            return

        self.assert_isinstance(value, "worker_ids", (list, tuple))

        self.assert_isinstance(value, "worker_ids", six.string_types, is_array=True)
        self._property_worker_ids = value

    @schema_property('from_date')
    def from_date(self):
        return self._property_from_date

    @from_date.setter
    def from_date(self, value):
        if value is None:
            self._property_from_date = None
            return

        self.assert_isinstance(value, "from_date", six.integer_types + (float,))
        self._property_from_date = value

    @schema_property('to_date')
    def to_date(self):
        return self._property_to_date

    @to_date.setter
    def to_date(self, value):
        if value is None:
            self._property_to_date = None
            return

        self.assert_isinstance(value, "to_date", six.integer_types + (float,))
        self._property_to_date = value

    @schema_property('interval')
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

    @schema_property('items')
    def items(self):
        return self._property_items

    @items.setter
    def items(self, value):
        if value is None:
            self._property_items = None
            return

        self.assert_isinstance(value, "items", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [StatItem.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "items", StatItem, is_array=True)
        self._property_items = value

    @schema_property('split_by_variant')
    def split_by_variant(self):
        return self._property_split_by_variant

    @split_by_variant.setter
    def split_by_variant(self, value):
        if value is None:
            self._property_split_by_variant = None
            return

        self.assert_isinstance(value, "split_by_variant", (bool,))
        self._property_split_by_variant = value


class GetStatsResponse(Response):
    """
    Response of workers.get_stats endpoint.

    :param workers: List of the requested workers with their statistics
    :type workers: Sequence[WorkerStats]
    """
    _service = "workers"
    _action = "get_stats"
    _version = "2.4"

    _schema = {
        'definitions': {
            'aggregation_stats': {
                'properties': {
                    'aggregation': {
                        'oneOf': [
                            {'$ref': '#/definitions/aggregation_type'},
                            {'type': 'null'},
                        ],
                    },
                    'values': {
                        'description': 'List of values corresponding to the dates in metric statistics',
                        'items': {'type': 'number'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'aggregation_type': {
                'description': 'Metric aggregation type',
                'enum': ['avg', 'min', 'max'],
                'type': 'string',
            },
            'metric_stats': {
                'properties': {
                    'dates': {
                        'description': 'List of timestamps (in seconds from epoch) in the acceding order. The timestamps are separated by the requested interval. Timestamps where no workers activity was recorded are omitted.',
                        'items': {'type': 'integer'},
                        'type': ['array', 'null'],
                    },
                    'metric': {
                        'description': 'Name of the metric (cpu_usage, memory_used etc.)',
                        'type': ['string', 'null'],
                    },
                    'stats': {
                        'description': 'Statistics data by type',
                        'items': {'$ref': '#/definitions/aggregation_stats'},
                        'type': ['array', 'null'],
                    },
                    'variant': {
                        'description': "Name of the metric component. Set only if 'split_by_variant' was set in the request",
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'worker_stats': {
                'properties': {
                    'metrics': {
                        'description': 'List of the metrics statistics for the worker',
                        'items': {'$ref': '#/definitions/metric_stats'},
                        'type': ['array', 'null'],
                    },
                    'worker': {
                        'description': 'ID of the worker',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
        },
        'properties': {
            'workers': {
                'description': 'List of the requested workers with their statistics',
                'items': {'$ref': '#/definitions/worker_stats'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }

    def __init__(
            self, workers=None, **kwargs):
        super(GetStatsResponse, self).__init__(**kwargs)
        self.workers = workers

    @schema_property('workers')
    def workers(self):
        return self._property_workers

    @workers.setter
    def workers(self, value):
        if value is None:
            self._property_workers = None
            return

        self.assert_isinstance(value, "workers", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [WorkerStats.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "workers", WorkerStats, is_array=True)
        self._property_workers = value


class RegisterRequest(Request):
    """
    Register a worker in the system. Called by the Worker Daemon.

    :param worker: Worker id. Must be unique in company.
    :type worker: str
    :param timeout: Registration timeout in seconds. If timeout seconds have passed
        since the worker's last call to register or status_report, the worker is
        automatically removed from the list of registered workers.
    :type timeout: int
    :param queues: List of queue IDs on which the worker is listening.
    :type queues: Sequence[str]
    """

    _service = "workers"
    _action = "register"
    _version = "2.4"
    _schema = {
        'definitions': {},
        'properties': {
            'queues': {
                'description': 'List of queue IDs on which the worker is listening.',
                'items': {'type': 'string'},
                'type': 'array',
            },
            'timeout': {
                'default': 600,
                'description': "Registration timeout in seconds. If timeout seconds have passed since the worker's last call to register or status_report, the worker is automatically removed from the list of registered workers.",
                'type': 'integer',
            },
            'worker': {
                'description': 'Worker id. Must be unique in company.',
                'type': 'string',
            },
        },
        'required': ['worker'],
        'type': 'object',
    }

    def __init__(
            self, worker, timeout=600, queues=None, **kwargs):
        super(RegisterRequest, self).__init__(**kwargs)
        self.worker = worker
        self.timeout = timeout
        self.queues = queues

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

    @schema_property('timeout')
    def timeout(self):
        return self._property_timeout

    @timeout.setter
    def timeout(self, value):
        if value is None:
            self._property_timeout = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "timeout", six.integer_types)
        self._property_timeout = value

    @schema_property('queues')
    def queues(self):
        return self._property_queues

    @queues.setter
    def queues(self, value):
        if value is None:
            self._property_queues = None
            return

        self.assert_isinstance(value, "queues", (list, tuple))

        self.assert_isinstance(value, "queues", six.string_types, is_array=True)
        self._property_queues = value


class RegisterResponse(Response):
    """
    Response of workers.register endpoint.

    """
    _service = "workers"
    _action = "register"
    _version = "2.4"

    _schema = {'definitions': {}, 'properties': {}, 'type': 'object'}


class StatusReportRequest(Request):
    """
    Called periodically by the worker daemon to report machine status

    :param worker: Worker id.
    :type worker: str
    :param task: ID of a task currently being run by the worker. If no task is
        sent, the worker's task field will be cleared.
    :type task: str
    :param queue: ID of the queue from which task was received. If no queue is
        sent, the worker's queue field will be cleared.
    :type queue: str
    :param queues: List of queue IDs on which the worker is listening. If null, the
        worker's queues list will not be updated.
    :type queues: Sequence[str]
    :param timestamp: UNIX time in seconds since epoch.
    :type timestamp: int
    :param machine_stats: The machine statistics.
    :type machine_stats: MachineStats
    """

    _service = "workers"
    _action = "status_report"
    _version = "2.4"
    _schema = {
        'definitions': {
            'machine_stats': {
                'properties': {
                    'cpu_temperature': {
                        'description': 'CPU temperature',
                        'items': {'type': 'number'},
                        'type': ['array', 'null'],
                    },
                    'cpu_usage': {
                        'description': 'Average CPU usage per core',
                        'items': {'type': 'number'},
                        'type': ['array', 'null'],
                    },
                    'disk_free_home': {
                        'description': 'Mbytes free space of /home drive',
                        'type': ['integer', 'null'],
                    },
                    'disk_free_temp': {
                        'description': 'Mbytes free space of /tmp drive',
                        'type': ['integer', 'null'],
                    },
                    'disk_read': {
                        'description': 'Mbytes read per second',
                        'type': ['integer', 'null'],
                    },
                    'disk_write': {
                        'description': 'Mbytes write per second',
                        'type': ['integer', 'null'],
                    },
                    'gpu_memory_free': {
                        'description': 'GPU free memory MBs',
                        'items': {'type': 'integer'},
                        'type': ['array', 'null'],
                    },
                    'gpu_memory_used': {
                        'description': 'GPU used memory MBs',
                        'items': {'type': 'integer'},
                        'type': ['array', 'null'],
                    },
                    'gpu_temperature': {
                        'description': 'GPU temperature',
                        'items': {'type': 'number'},
                        'type': ['array', 'null'],
                    },
                    'gpu_usage': {
                        'description': 'Average GPU usage per GPU card',
                        'items': {'type': 'number'},
                        'type': ['array', 'null'],
                    },
                    'memory_free': {
                        'description': 'Free memory MBs',
                        'type': ['integer', 'null'],
                    },
                    'memory_used': {
                        'description': 'Used memory MBs',
                        'type': ['integer', 'null'],
                    },
                    'network_rx': {
                        'description': 'Mbytes per second',
                        'type': ['integer', 'null'],
                    },
                    'network_tx': {
                        'description': 'Mbytes per second',
                        'type': ['integer', 'null'],
                    },
                },
                'type': 'object',
            },
        },
        'properties': {
            'machine_stats': {
                '$ref': '#/definitions/machine_stats',
                'description': 'The machine statistics.',
            },
            'queue': {
                'description': "ID of the queue from which task was received. If no queue is sent, the worker's queue field will be cleared.",
                'type': 'string',
            },
            'queues': {
                'description': "List of queue IDs on which the worker is listening. If null, the worker's queues list will not be updated.",
                'items': {'type': 'string'},
                'type': 'array',
            },
            'task': {
                'description': "ID of a task currently being run by the worker. If no task is sent, the worker's task field will be cleared.",
                'type': 'string',
            },
            'timestamp': {
                'description': 'UNIX time in seconds since epoch.',
                'type': 'integer',
            },
            'worker': {'description': 'Worker id.', 'type': 'string'},
        },
        'required': ['worker', 'timestamp'],
        'type': 'object',
    }

    def __init__(
            self, worker, timestamp, task=None, queue=None, queues=None, machine_stats=None, **kwargs):
        super(StatusReportRequest, self).__init__(**kwargs)
        self.worker = worker
        self.task = task
        self.queue = queue
        self.queues = queues
        self.timestamp = timestamp
        self.machine_stats = machine_stats

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

    @schema_property('queue')
    def queue(self):
        return self._property_queue

    @queue.setter
    def queue(self, value):
        if value is None:
            self._property_queue = None
            return

        self.assert_isinstance(value, "queue", six.string_types)
        self._property_queue = value

    @schema_property('queues')
    def queues(self):
        return self._property_queues

    @queues.setter
    def queues(self, value):
        if value is None:
            self._property_queues = None
            return

        self.assert_isinstance(value, "queues", (list, tuple))

        self.assert_isinstance(value, "queues", six.string_types, is_array=True)
        self._property_queues = value

    @schema_property('timestamp')
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "timestamp", six.integer_types)
        self._property_timestamp = value

    @schema_property('machine_stats')
    def machine_stats(self):
        return self._property_machine_stats

    @machine_stats.setter
    def machine_stats(self, value):
        if value is None:
            self._property_machine_stats = None
            return
        if isinstance(value, dict):
            value = MachineStats.from_dict(value)
        else:
            self.assert_isinstance(value, "machine_stats", MachineStats)
        self._property_machine_stats = value


class StatusReportResponse(Response):
    """
    Response of workers.status_report endpoint.

    """
    _service = "workers"
    _action = "status_report"
    _version = "2.4"

    _schema = {'definitions': {}, 'properties': {}, 'type': 'object'}


class UnregisterRequest(Request):
    """
    Unregister a worker in the system. Called by the Worker Daemon.

    :param worker: Worker id. Must be unique in company.
    :type worker: str
    """

    _service = "workers"
    _action = "unregister"
    _version = "2.4"
    _schema = {
        'definitions': {},
        'properties': {
            'worker': {
                'description': 'Worker id. Must be unique in company.',
                'type': 'string',
            },
        },
        'required': ['worker'],
        'type': 'object',
    }

    def __init__(
            self, worker, **kwargs):
        super(UnregisterRequest, self).__init__(**kwargs)
        self.worker = worker

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


class UnregisterResponse(Response):
    """
    Response of workers.unregister endpoint.

    """
    _service = "workers"
    _action = "unregister"
    _version = "2.4"

    _schema = {'definitions': {}, 'properties': {}, 'type': 'object'}


response_mapping = {
    GetAllRequest: GetAllResponse,
    RegisterRequest: RegisterResponse,
    UnregisterRequest: UnregisterResponse,
    StatusReportRequest: StatusReportResponse,
    GetMetricKeysRequest: GetMetricKeysResponse,
    GetStatsRequest: GetStatsResponse,
    GetActivityReportRequest: GetActivityReportResponse,
}
