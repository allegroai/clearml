"""
projects service

Provides support for defining Projects containing Tasks, Models and Dataset Versions.
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


class MultiFieldPatternData(NonStrictDataModel):
    """
    :param pattern: Pattern string (regex)
    :type pattern: str
    :param fields: List of field names
    :type fields: Sequence[str]
    """

    _schema = {
        "properties": {
            "fields": {
                "description": "List of field names",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "pattern": {
                "description": "Pattern string (regex)",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, pattern=None, fields=None, **kwargs):
        super(MultiFieldPatternData, self).__init__(**kwargs)
        self.pattern = pattern
        self.fields = fields

    @schema_property("pattern")
    def pattern(self):
        return self._property_pattern

    @pattern.setter
    def pattern(self, value):
        if value is None:
            self._property_pattern = None
            return

        self.assert_isinstance(value, "pattern", six.string_types)
        self._property_pattern = value

    @schema_property("fields")
    def fields(self):
        return self._property_fields

    @fields.setter
    def fields(self, value):
        if value is None:
            self._property_fields = None
            return

        self.assert_isinstance(value, "fields", (list, tuple))

        self.assert_isinstance(value, "fields", six.string_types, is_array=True)
        self._property_fields = value


class Project(NonStrictDataModel):
    """
    :param id: Project id
    :type id: str
    :param name: Project name
    :type name: str
    :param basename: Project base name
    :type basename: str
    :param description: Project description
    :type description: str
    :param user: Associated user id
    :type user: str
    :param company: Company id
    :type company: str
    :param created: Creation time
    :type created: datetime.datetime
    :param last_update: Last project update time. Reflects the last time the
        project metadata was changed or a task in this project has changed status
    :type last_update: datetime.datetime
    :param tags: User-defined tags
    :type tags: Sequence[str]
    :param system_tags: System tags. This field is reserved for system use, please
        don't use it.
    :type system_tags: Sequence[str]
    :param default_output_destination: The default output destination URL for new
        tasks under this project
    :type default_output_destination: str
    """

    _schema = {
        "properties": {
            "basename": {
                "description": "Project base name",
                "type": ["string", "null"],
            },
            "company": {"description": "Company id", "type": ["string", "null"]},
            "created": {
                "description": "Creation time",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "default_output_destination": {
                "description": "The default output destination URL for new tasks under this project",
                "type": ["string", "null"],
            },
            "description": {
                "description": "Project description",
                "type": ["string", "null"],
            },
            "id": {"description": "Project id", "type": ["string", "null"]},
            "last_update": {
                "description": (
                    "Last project update time. Reflects the last time the project metadata was changed or a task in"
                    " this project has changed status"
                ),
                "format": "date-time",
                "type": ["string", "null"],
            },
            "name": {"description": "Project name", "type": ["string", "null"]},
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
        basename=None,
        description=None,
        user=None,
        company=None,
        created=None,
        last_update=None,
        tags=None,
        system_tags=None,
        default_output_destination=None,
        **kwargs
    ):
        super(Project, self).__init__(**kwargs)
        self.id = id
        self.name = name
        self.basename = basename
        self.description = description
        self.user = user
        self.company = company
        self.created = created
        self.last_update = last_update
        self.tags = tags
        self.system_tags = system_tags
        self.default_output_destination = default_output_destination

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

    @schema_property("basename")
    def basename(self):
        return self._property_basename

    @basename.setter
    def basename(self, value):
        if value is None:
            self._property_basename = None
            return

        self.assert_isinstance(value, "basename", six.string_types)
        self._property_basename = value

    @schema_property("description")
    def description(self):
        return self._property_description

    @description.setter
    def description(self, value):
        if value is None:
            self._property_description = None
            return

        self.assert_isinstance(value, "description", six.string_types)
        self._property_description = value

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

    @schema_property("last_update")
    def last_update(self):
        return self._property_last_update

    @last_update.setter
    def last_update(self, value):
        if value is None:
            self._property_last_update = None
            return

        self.assert_isinstance(value, "last_update", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_last_update = value

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

    @schema_property("default_output_destination")
    def default_output_destination(self):
        return self._property_default_output_destination

    @default_output_destination.setter
    def default_output_destination(self, value):
        if value is None:
            self._property_default_output_destination = None
            return

        self.assert_isinstance(value, "default_output_destination", six.string_types)
        self._property_default_output_destination = value


class StatsStatusCount(NonStrictDataModel):
    """
    :param total_runtime: Total run time of all tasks in project (in seconds)
    :type total_runtime: int
    :param total_tasks: Number of tasks
    :type total_tasks: int
    :param completed_tasks_24h: Number of tasks completed in the last 24 hours
    :type completed_tasks_24h: int
    :param last_task_run: The most recent started time of a task
    :type last_task_run: int
    :param status_count: Status counts
    :type status_count: dict
    """

    _schema = {
        "properties": {
            "completed_tasks_24h": {
                "description": "Number of tasks completed in the last 24 hours",
                "type": ["integer", "null"],
            },
            "last_task_run": {
                "description": "The most recent started time of a task",
                "type": ["integer", "null"],
            },
            "status_count": {
                "description": "Status counts",
                "properties": {
                    "closed": {
                        "description": "Number of 'closed' tasks in project",
                        "type": "integer",
                    },
                    "completed": {
                        "description": "Number of 'completed' tasks in project",
                        "type": "integer",
                    },
                    "created": {
                        "description": "Number of 'created' tasks in project",
                        "type": "integer",
                    },
                    "failed": {
                        "description": "Number of 'failed' tasks in project",
                        "type": "integer",
                    },
                    "in_progress": {
                        "description": "Number of 'in_progress' tasks in project",
                        "type": "integer",
                    },
                    "published": {
                        "description": "Number of 'published' tasks in project",
                        "type": "integer",
                    },
                    "queued": {
                        "description": "Number of 'queued' tasks in project",
                        "type": "integer",
                    },
                    "stopped": {
                        "description": "Number of 'stopped' tasks in project",
                        "type": "integer",
                    },
                    "unknown": {
                        "description": "Number of 'unknown' tasks in project",
                        "type": "integer",
                    },
                },
                "type": ["object", "null"],
            },
            "total_runtime": {
                "description": "Total run time of all tasks in project (in seconds)",
                "type": ["integer", "null"],
            },
            "total_tasks": {
                "description": "Number of tasks",
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(
        self,
        total_runtime=None,
        total_tasks=None,
        completed_tasks_24h=None,
        last_task_run=None,
        status_count=None,
        **kwargs
    ):
        super(StatsStatusCount, self).__init__(**kwargs)
        self.total_runtime = total_runtime
        self.total_tasks = total_tasks
        self.completed_tasks_24h = completed_tasks_24h
        self.last_task_run = last_task_run
        self.status_count = status_count

    @schema_property("total_runtime")
    def total_runtime(self):
        return self._property_total_runtime

    @total_runtime.setter
    def total_runtime(self, value):
        if value is None:
            self._property_total_runtime = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "total_runtime", six.integer_types)
        self._property_total_runtime = value

    @schema_property("total_tasks")
    def total_tasks(self):
        return self._property_total_tasks

    @total_tasks.setter
    def total_tasks(self, value):
        if value is None:
            self._property_total_tasks = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "total_tasks", six.integer_types)
        self._property_total_tasks = value

    @schema_property("completed_tasks_24h")
    def completed_tasks_24h(self):
        return self._property_completed_tasks_24h

    @completed_tasks_24h.setter
    def completed_tasks_24h(self, value):
        if value is None:
            self._property_completed_tasks_24h = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "completed_tasks_24h", six.integer_types)
        self._property_completed_tasks_24h = value

    @schema_property("last_task_run")
    def last_task_run(self):
        return self._property_last_task_run

    @last_task_run.setter
    def last_task_run(self, value):
        if value is None:
            self._property_last_task_run = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "last_task_run", six.integer_types)
        self._property_last_task_run = value

    @schema_property("status_count")
    def status_count(self):
        return self._property_status_count

    @status_count.setter
    def status_count(self, value):
        if value is None:
            self._property_status_count = None
            return

        self.assert_isinstance(value, "status_count", (dict,))
        self._property_status_count = value


class Stats(NonStrictDataModel):
    """
    :param active: Stats for active tasks
    :type active: StatsStatusCount
    :param archived: Stats for archived tasks
    :type archived: StatsStatusCount
    """

    _schema = {
        "properties": {
            "active": {
                "description": "Stats for active tasks",
                "oneOf": [
                    {"$ref": "#/definitions/stats_status_count"},
                    {"type": "null"},
                ],
            },
            "archived": {
                "description": "Stats for archived tasks",
                "oneOf": [
                    {"$ref": "#/definitions/stats_status_count"},
                    {"type": "null"},
                ],
            },
        },
        "type": "object",
    }

    def __init__(self, active=None, archived=None, **kwargs):
        super(Stats, self).__init__(**kwargs)
        self.active = active
        self.archived = archived

    @schema_property("active")
    def active(self):
        return self._property_active

    @active.setter
    def active(self, value):
        if value is None:
            self._property_active = None
            return
        if isinstance(value, dict):
            value = StatsStatusCount.from_dict(value)
        else:
            self.assert_isinstance(value, "active", StatsStatusCount)
        self._property_active = value

    @schema_property("archived")
    def archived(self):
        return self._property_archived

    @archived.setter
    def archived(self, value):
        if value is None:
            self._property_archived = None
            return
        if isinstance(value, dict):
            value = StatsStatusCount.from_dict(value)
        else:
            self.assert_isinstance(value, "archived", StatsStatusCount)
        self._property_archived = value


class ProjectsGetAllResponseSingle(NonStrictDataModel):
    """
    :param id: Project id
    :type id: str
    :param name: Project name
    :type name: str
    :param basename: Project base name
    :type basename: str
    :param description: Project description
    :type description: str
    :param user: Associated user id
    :type user: str
    :param company: Company id
    :type company: str
    :param created: Creation time
    :type created: datetime.datetime
    :param last_update: Last update time
    :type last_update: datetime.datetime
    :param tags: User-defined tags
    :type tags: Sequence[str]
    :param system_tags: System tags. This field is reserved for system use, please don't use it.
    :type system_tags: Sequence[str]
    :param default_output_destination: The default output destination URL for new tasks under this project
    :type default_output_destination: str
    :param stats: Additional project stats
    :type stats: Stats
    :param sub_projects: The list of sub projects
    :type sub_projects: Sequence[dict]
    :param own_tasks: The amount of tasks under this project (without children projects).
        Returned if 'check_own_contents' flag is set in the request
    :type own_tasks: int
    :param own_models: The amount of models under this project (without children projects).
        Returned if 'check_own_contents' flag is set in the request
    :type own_models: int
    :param dataset_stats: Project dataset statistics
    :type dataset_stats: dict
    """

    _schema = {
        "properties": {
            "basename": {
                "description": "Project base name",
                "type": ["string", "null"],
            },
            "company": {"description": "Company id", "type": ["string", "null"]},
            "created": {
                "description": "Creation time",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "dataset_stats": {
                "description": "Project dataset statistics",
                "properties": {
                    "file_count": {
                        "description": "The number of files stored in the dataset",
                        "type": "integer",
                    },
                    "total_size": {
                        "description": "The total dataset size in bytes",
                        "type": "integer",
                    },
                },
                "type": ["object", "null"],
            },
            "default_output_destination": {
                "description": "The default output destination URL for new tasks under this project",
                "type": ["string", "null"],
            },
            "description": {
                "description": "Project description",
                "type": ["string", "null"],
            },
            "id": {"description": "Project id", "type": ["string", "null"]},
            "last_update": {
                "description": "Last update time",
                "format": "date-time",
                "type": ["string", "null"],
            },
            "name": {"description": "Project name", "type": ["string", "null"]},
            "own_models": {
                "description": (
                    "The amount of models under this project (without children projects). Returned if "
                    "'check_own_contents' flag is set in the request"
                ),
                "type": ["integer", "null"],
            },
            "own_tasks": {
                "description": (
                    "The amount of tasks under this project (without children projects). Returned if "
                    "'check_own_contents' flag is set in the request"
                ),
                "type": ["integer", "null"],
            },
            "stats": {
                "description": "Additional project stats",
                "oneOf": [{"$ref": "#/definitions/stats"}, {"type": "null"}],
            },
            "sub_projects": {
                "description": "The list of sub projects",
                "items": {
                    "properties": {
                        "id": {"description": "Subproject ID", "type": "string"},
                        "name": {"description": "Subproject name", "type": "string"},
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            },
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
        basename=None,
        description=None,
        user=None,
        company=None,
        created=None,
        last_update=None,
        tags=None,
        system_tags=None,
        default_output_destination=None,
        stats=None,
        sub_projects=None,
        own_tasks=None,
        own_models=None,
        dataset_stats=None,
        **kwargs
    ):
        super(ProjectsGetAllResponseSingle, self).__init__(**kwargs)
        self.id = id
        self.name = name
        self.basename = basename
        self.description = description
        self.user = user
        self.company = company
        self.created = created
        self.last_update = last_update
        self.tags = tags
        self.system_tags = system_tags
        self.default_output_destination = default_output_destination
        self.stats = stats
        self.sub_projects = sub_projects
        self.own_tasks = own_tasks
        self.own_models = own_models
        self.dataset_stats = dataset_stats

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

    @schema_property("basename")
    def basename(self):
        return self._property_basename

    @basename.setter
    def basename(self, value):
        if value is None:
            self._property_basename = None
            return

        self.assert_isinstance(value, "basename", six.string_types)
        self._property_basename = value

    @schema_property("description")
    def description(self):
        return self._property_description

    @description.setter
    def description(self, value):
        if value is None:
            self._property_description = None
            return

        self.assert_isinstance(value, "description", six.string_types)
        self._property_description = value

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

    @schema_property("last_update")
    def last_update(self):
        return self._property_last_update

    @last_update.setter
    def last_update(self, value):
        if value is None:
            self._property_last_update = None
            return

        self.assert_isinstance(value, "last_update", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_last_update = value

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

    @schema_property("default_output_destination")
    def default_output_destination(self):
        return self._property_default_output_destination

    @default_output_destination.setter
    def default_output_destination(self, value):
        if value is None:
            self._property_default_output_destination = None
            return

        self.assert_isinstance(value, "default_output_destination", six.string_types)
        self._property_default_output_destination = value

    @schema_property("stats")
    def stats(self):
        return self._property_stats

    @stats.setter
    def stats(self, value):
        if value is None:
            self._property_stats = None
            return
        if isinstance(value, dict):
            value = Stats.from_dict(value)
        else:
            self.assert_isinstance(value, "stats", Stats)
        self._property_stats = value

    @schema_property("sub_projects")
    def sub_projects(self):
        return self._property_sub_projects

    @sub_projects.setter
    def sub_projects(self, value):
        if value is None:
            self._property_sub_projects = None
            return

        self.assert_isinstance(value, "sub_projects", (list, tuple))

        self.assert_isinstance(value, "sub_projects", (dict,), is_array=True)
        self._property_sub_projects = value

    @schema_property("own_tasks")
    def own_tasks(self):
        return self._property_own_tasks

    @own_tasks.setter
    def own_tasks(self, value):
        if value is None:
            self._property_own_tasks = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "own_tasks", six.integer_types)
        self._property_own_tasks = value

    @schema_property("own_models")
    def own_models(self):
        return self._property_own_models

    @own_models.setter
    def own_models(self, value):
        if value is None:
            self._property_own_models = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "own_models", six.integer_types)
        self._property_own_models = value

    @schema_property("dataset_stats")
    def dataset_stats(self):
        return self._property_dataset_stats

    @dataset_stats.setter
    def dataset_stats(self, value):
        if value is None:
            self._property_dataset_stats = None
            return

        self.assert_isinstance(value, "dataset_stats", (dict,))
        self._property_dataset_stats = value


class MetricVariantResult(NonStrictDataModel):
    """
    :param metric: Metric name
    :type metric: str
    :param metric_hash: Metric name hash. Used instead of the metric name when
        categorizing last metrics events in task objects.
    :type metric_hash: str
    :param variant: Variant name
    :type variant: str
    :param variant_hash: Variant name hash. Used instead of the variant name when
        categorizing last metrics events in task objects.
    :type variant_hash: str
    """

    _schema = {
        "properties": {
            "metric": {"description": "Metric name", "type": ["string", "null"]},
            "metric_hash": {
                "description": (
                    "Metric name hash. Used instead of the metric name when categorizing "
                    " last metrics events in task objects."
                ),
                "type": ["string", "null"],
            },
            "variant": {"description": "Variant name", "type": ["string", "null"]},
            "variant_hash": {
                "description": (
                    "Variant name hash. Used instead of the variant name when categorizing "
                    "last metrics events in task objects."
                ),
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, metric=None, metric_hash=None, variant=None, variant_hash=None, **kwargs):
        super(MetricVariantResult, self).__init__(**kwargs)
        self.metric = metric
        self.metric_hash = metric_hash
        self.variant = variant
        self.variant_hash = variant_hash

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

    @schema_property("metric_hash")
    def metric_hash(self):
        return self._property_metric_hash

    @metric_hash.setter
    def metric_hash(self, value):
        if value is None:
            self._property_metric_hash = None
            return

        self.assert_isinstance(value, "metric_hash", six.string_types)
        self._property_metric_hash = value

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

    @schema_property("variant_hash")
    def variant_hash(self):
        return self._property_variant_hash

    @variant_hash.setter
    def variant_hash(self, value):
        if value is None:
            self._property_variant_hash = None
            return

        self.assert_isinstance(value, "variant_hash", six.string_types)
        self._property_variant_hash = value


class Urls(NonStrictDataModel):
    """
    :param model_urls:
    :type model_urls: Sequence[str]
    :param event_urls:
    :type event_urls: Sequence[str]
    :param artifact_urls:
    :type artifact_urls: Sequence[str]
    """

    _schema = {
        "properties": {
            "artifact_urls": {"items": {"type": "string"}, "type": ["array", "null"]},
            "event_urls": {"items": {"type": "string"}, "type": ["array", "null"]},
            "model_urls": {"items": {"type": "string"}, "type": ["array", "null"]},
        },
        "type": "object",
    }

    def __init__(self, model_urls=None, event_urls=None, artifact_urls=None, **kwargs):
        super(Urls, self).__init__(**kwargs)
        self.model_urls = model_urls
        self.event_urls = event_urls
        self.artifact_urls = artifact_urls

    @schema_property("model_urls")
    def model_urls(self):
        return self._property_model_urls

    @model_urls.setter
    def model_urls(self, value):
        if value is None:
            self._property_model_urls = None
            return

        self.assert_isinstance(value, "model_urls", (list, tuple))

        self.assert_isinstance(value, "model_urls", six.string_types, is_array=True)
        self._property_model_urls = value

    @schema_property("event_urls")
    def event_urls(self):
        return self._property_event_urls

    @event_urls.setter
    def event_urls(self, value):
        if value is None:
            self._property_event_urls = None
            return

        self.assert_isinstance(value, "event_urls", (list, tuple))

        self.assert_isinstance(value, "event_urls", six.string_types, is_array=True)
        self._property_event_urls = value

    @schema_property("artifact_urls")
    def artifact_urls(self):
        return self._property_artifact_urls

    @artifact_urls.setter
    def artifact_urls(self, value):
        if value is None:
            self._property_artifact_urls = None
            return

        self.assert_isinstance(value, "artifact_urls", (list, tuple))

        self.assert_isinstance(value, "artifact_urls", six.string_types, is_array=True)
        self._property_artifact_urls = value


class CreateRequest(Request):
    """
    Create a new project

    :param name: Project name Unique within the company.
    :type name: str
    :param description: Project description.
    :type description: str
    :param tags: User-defined tags
    :type tags: Sequence[str]
    :param system_tags: System tags. This field is reserved for system use, please don't use it.
    :type system_tags: Sequence[str]
    :param default_output_destination: The default output destination URL for new tasks under this project
    :type default_output_destination: str
    """

    _service = "projects"
    _action = "create"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "default_output_destination": {
                "description": "The default output destination URL for new tasks under this project",
                "type": "string",
            },
            "description": {"description": "Project description.", "type": "string"},
            "name": {
                "description": "Project name Unique within the company.",
                "type": "string",
            },
            "system_tags": {
                "description": "System tags. This field is reserved for system use, please don't use it.",
                "items": {"type": "string"},
                "type": "array",
            },
            "tags": {
                "description": "User-defined tags",
                "items": {"type": "string"},
                "type": "array",
            },
        },
        "required": ["name"],
        "type": "object",
    }

    def __init__(self, name, description=None, tags=None, system_tags=None, default_output_destination=None, **kwargs):
        super(CreateRequest, self).__init__(**kwargs)
        self.name = name
        self.description = description
        self.tags = tags
        self.system_tags = system_tags
        self.default_output_destination = default_output_destination

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

    @schema_property("description")
    def description(self):
        return self._property_description

    @description.setter
    def description(self, value):
        if value is None:
            self._property_description = None
            return

        self.assert_isinstance(value, "description", six.string_types)
        self._property_description = value

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

    @schema_property("default_output_destination")
    def default_output_destination(self):
        return self._property_default_output_destination

    @default_output_destination.setter
    def default_output_destination(self, value):
        if value is None:
            self._property_default_output_destination = None
            return

        self.assert_isinstance(value, "default_output_destination", six.string_types)
        self._property_default_output_destination = value


class CreateResponse(Response):
    """
    Response of projects.create endpoint.

    :param id: Project id
    :type id: str
    """

    _service = "projects"
    _action = "create"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {"id": {"description": "Project id", "type": ["string", "null"]}},
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
    Deletes a project

    :param project: Project ID
    :type project: str
    :param force: If not true, fails if project has tasks. If true, and project has tasks, they will be unassigned
    :type force: bool
    :param delete_contents: If set to 'true' then the project tasks and models will be deleted.
        Otherwise their project property will be unassigned. Default value is 'false'
    :type delete_contents: bool
    """

    _service = "projects"
    _action = "delete"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "delete_contents": {
                "description": (
                    "If set to 'true' then the project tasks, models and dataviews will be deleted. Otherwise their"
                    " project property will be unassigned. Default value is 'false'"
                ),
                "type": "boolean",
            },
            "force": {
                "default": False,
                "description": (
                    "If not true, fails if project has tasks. If true, and project has tasks, they will be unassigned"
                ),
                "type": "boolean",
            },
            "project": {"description": "Project ID", "type": "string"},
        },
        "required": ["project"],
        "type": "object",
    }

    def __init__(self, project, force=False, delete_contents=None, **kwargs):
        super(DeleteRequest, self).__init__(**kwargs)
        self.project = project
        self.force = force
        self.delete_contents = delete_contents

    @schema_property("project")
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return

        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

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

    @schema_property("delete_contents")
    def delete_contents(self):
        return self._property_delete_contents

    @delete_contents.setter
    def delete_contents(self, value):
        if value is None:
            self._property_delete_contents = None
            return

        self.assert_isinstance(value, "delete_contents", (bool,))
        self._property_delete_contents = value


class DeleteResponse(Response):
    """
    Response of projects.delete endpoint.

    :param deleted: Number of projects deleted (0 or 1)
    :type deleted: int
    :param disassociated_tasks: Number of tasks disassociated from the deleted project
    :type disassociated_tasks: int
    :param urls: The urls of the files that were uploaded by the project tasks and models.
        Returned if the 'delete_contents' was set to 'true'
    :type urls: Urls
    :param deleted_models: Number of models deleted
    :type deleted_models: int
    :param deleted_tasks: Number of tasks deleted
    :type deleted_tasks: int
    """

    _service = "projects"
    _action = "delete"
    _version = "2.20"

    _schema = {
        "definitions": {
            "urls": {
                "properties": {
                    "artifact_urls": {
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "event_urls": {
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "model_urls": {
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "deleted": {
                "description": "Number of projects deleted (0 or 1)",
                "type": ["integer", "null"],
            },
            "deleted_models": {
                "description": "Number of models deleted",
                "type": ["integer", "null"],
            },
            "deleted_tasks": {
                "description": "Number of tasks deleted",
                "type": ["integer", "null"],
            },
            "disassociated_tasks": {
                "description": "Number of tasks disassociated from the deleted project",
                "type": ["integer", "null"],
            },
            "urls": {
                "description": (
                    "The urls of the files that were uploaded by the project tasks and models. Returned if the"
                    " 'delete_contents' was set to 'true'"
                ),
                "oneOf": [{"$ref": "#/definitions/urls"}, {"type": "null"}],
            },
        },
        "type": "object",
    }

    def __init__(
        self, deleted=None, disassociated_tasks=None, urls=None, deleted_models=None, deleted_tasks=None, **kwargs
    ):
        super(DeleteResponse, self).__init__(**kwargs)
        self.deleted = deleted
        self.disassociated_tasks = disassociated_tasks
        self.urls = urls
        self.deleted_models = deleted_models
        self.deleted_tasks = deleted_tasks

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

    @schema_property("disassociated_tasks")
    def disassociated_tasks(self):
        return self._property_disassociated_tasks

    @disassociated_tasks.setter
    def disassociated_tasks(self, value):
        if value is None:
            self._property_disassociated_tasks = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "disassociated_tasks", six.integer_types)
        self._property_disassociated_tasks = value

    @schema_property("urls")
    def urls(self):
        return self._property_urls

    @urls.setter
    def urls(self, value):
        if value is None:
            self._property_urls = None
            return
        if isinstance(value, dict):
            value = Urls.from_dict(value)
        else:
            self.assert_isinstance(value, "urls", Urls)
        self._property_urls = value

    @schema_property("deleted_models")
    def deleted_models(self):
        return self._property_deleted_models

    @deleted_models.setter
    def deleted_models(self, value):
        if value is None:
            self._property_deleted_models = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "deleted_models", six.integer_types)
        self._property_deleted_models = value

    @schema_property("deleted_tasks")
    def deleted_tasks(self):
        return self._property_deleted_tasks

    @deleted_tasks.setter
    def deleted_tasks(self, value):
        if value is None:
            self._property_deleted_tasks = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "deleted_tasks", six.integer_types)
        self._property_deleted_tasks = value


class GetAllRequest(Request):
    """
    Get all the company's projects and all public projects

    :param id: List of IDs to filter by
    :type id: Sequence[str]
    :param name: Get only projects whose name matches this pattern (python regular expression syntax)
    :type name: str
    :param basename: Project base name
    :type basename: str
    :param description: Get only projects whose description matches this pattern (python regular expression syntax)
    :type description: str
    :param tags: User-defined tags list used to filter results. Prepend '-' to tag name to indicate exclusion
    :type tags: Sequence[str]
    :param system_tags: System tags list used to filter results. Prepend '-' to system tag name to indicate exclusion
    :type system_tags: Sequence[str]
    :param order_by: List of field names to order by. When search_text is used, '@text_score' can be used as a field
        representing the text score of returned documents. Use '-' prefix to specify descending order.
        Optional, recommended when using page
    :type order_by: Sequence[str]
    :param page: Page number, returns a specific page out of the resulting list of projects
    :type page: int
    :param page_size: Page size, specifies the number of results returned in each page
        (last page may contain fewer results)
    :type page_size: int
    :param search_text: Free text search query
    :type search_text: str
    :param only_fields: List of document's field names (nesting is supported using '.', e.g. execution.model_labels).
        If provided, this list defines the query's projection (only these fields will be returned for each result entry)
    :type only_fields: Sequence[str]
    :param _all_: Multi-field pattern condition (all fields match pattern)
    :type _all_: MultiFieldPatternData
    :param _any_: Multi-field pattern condition (any field matches pattern)
    :type _any_: MultiFieldPatternData
    :param shallow_search: If set to 'true' then the search with the specified criteria is performed among top level
        projects only (or if parents specified, among the direct children of the these parents). Otherwise the search is
        performed among all the company projects (or among all of the descendants of
        the specified parents).
    :type shallow_search: bool
    :param search_hidden: If set to 'true' then hidden projects are included in the search results
    :type search_hidden: bool
    :param scroll_id: Scroll ID returned from the previos calls to get_all_ex
    :type scroll_id: str
    :param refresh_scroll: If set then all the data received with this scroll will be required
    :type refresh_scroll: bool
    :param size: The number of projects to retrieve
    :type size: int
    """

    _service = "projects"
    _action = "get_all"
    _version = "2.20"
    _schema = {
        "definitions": {
            "multi_field_pattern_data": {
                "properties": {
                    "fields": {
                        "description": "List of field names",
                        "items": {"type": "string"},
                        "type": ["array", "null"],
                    },
                    "pattern": {
                        "description": "Pattern string (regex)",
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "_all_": {
                "description": "Multi-field pattern condition (all fields match pattern)",
                "oneOf": [
                    {"$ref": "#/definitions/multi_field_pattern_data"},
                    {"type": "null"},
                ],
            },
            "_any_": {
                "description": "Multi-field pattern condition (any field matches pattern)",
                "oneOf": [
                    {"$ref": "#/definitions/multi_field_pattern_data"},
                    {"type": "null"},
                ],
            },
            "basename": {
                "description": "Project base name",
                "type": ["string", "null"],
            },
            "description": {
                "description": (
                    "Get only projects whose description matches this pattern (python regular expression syntax)"
                ),
                "type": ["string", "null"],
            },
            "id": {
                "description": "List of IDs to filter by",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "name": {
                "description": "Get only projects whose name matches this pattern (python regular expression syntax)",
                "type": ["string", "null"],
            },
            "only_fields": {
                "description": (
                    "List of document's field names (nesting is supported using '.', e.g. execution.model_labels). If"
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
                "description": "Page number, returns a specific page out of the resulting list of dataviews",
                "minimum": 0,
                "type": ["integer", "null"],
            },
            "page_size": {
                "description": (
                    "Page size, specifies the number of results returned in each page (last page may contain fewer"
                    " results)"
                ),
                "minimum": 1,
                "type": ["integer", "null"],
            },
            "refresh_scroll": {
                "description": "If set then all the data received with this scroll will be requeried",
                "type": ["boolean", "null"],
            },
            "scroll_id": {
                "description": "Scroll ID returned from the previos calls to get_all_ex",
                "type": ["string", "null"],
            },
            "search_hidden": {
                "default": False,
                "description": "If set to 'true' then hidden projects are included in the search results",
                "type": ["boolean", "null"],
            },
            "search_text": {
                "description": "Free text search query",
                "type": ["string", "null"],
            },
            "shallow_search": {
                "default": False,
                "description": (
                    "If set to 'true' then the search with the specified criteria is performed among top level projects"
                    " only (or if parents specified, among the direct children of the these parents). Otherwise the"
                    " search is performed among all the company projects (or among all of the descendants of the"
                    " specified parents)."
                ),
                "type": ["boolean", "null"],
            },
            "size": {
                "description": "The number of projects to retrieve",
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
        id=None,
        name=None,
        basename=None,
        description=None,
        tags=None,
        system_tags=None,
        order_by=None,
        page=None,
        page_size=None,
        search_text=None,
        only_fields=None,
        _all_=None,
        _any_=None,
        shallow_search=False,
        search_hidden=False,
        scroll_id=None,
        refresh_scroll=None,
        size=None,
        **kwargs
    ):
        super(GetAllRequest, self).__init__(**kwargs)
        self.id = id
        self.name = name
        self.basename = basename
        self.description = description
        self.tags = tags
        self.system_tags = system_tags
        self.order_by = order_by
        self.page = page
        self.page_size = page_size
        self.search_text = search_text
        self.only_fields = only_fields
        self._all_ = _all_
        self._any_ = _any_
        self.shallow_search = shallow_search
        self.search_hidden = search_hidden
        self.scroll_id = scroll_id
        self.refresh_scroll = refresh_scroll
        self.size = size

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

    @schema_property("basename")
    def basename(self):
        return self._property_basename

    @basename.setter
    def basename(self, value):
        if value is None:
            self._property_basename = None
            return

        self.assert_isinstance(value, "basename", six.string_types)
        self._property_basename = value

    @schema_property("description")
    def description(self):
        return self._property_description

    @description.setter
    def description(self, value):
        if value is None:
            self._property_description = None
            return

        self.assert_isinstance(value, "description", six.string_types)
        self._property_description = value

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

    @schema_property("_all_")
    def _all_(self):
        return self._property__all_

    @_all_.setter
    def _all_(self, value):
        if value is None:
            self._property__all_ = None
            return
        if isinstance(value, dict):
            value = MultiFieldPatternData.from_dict(value)
        else:
            self.assert_isinstance(value, "_all_", MultiFieldPatternData)
        self._property__all_ = value

    @schema_property("_any_")
    def _any_(self):
        return self._property__any_

    @_any_.setter
    def _any_(self, value):
        if value is None:
            self._property__any_ = None
            return
        if isinstance(value, dict):
            value = MultiFieldPatternData.from_dict(value)
        else:
            self.assert_isinstance(value, "_any_", MultiFieldPatternData)
        self._property__any_ = value

    @schema_property("shallow_search")
    def shallow_search(self):
        return self._property_shallow_search

    @shallow_search.setter
    def shallow_search(self, value):
        if value is None:
            self._property_shallow_search = None
            return

        self.assert_isinstance(value, "shallow_search", (bool,))
        self._property_shallow_search = value

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


class GetAllResponse(Response):
    """
    Response of projects.get_all endpoint.

    :param projects: Projects list
    :type projects: Sequence[ProjectsGetAllResponseSingle]
    :param scroll_id: Scroll ID that can be used with the next calls to get_all_ex to retrieve more data
    :type scroll_id: str
    """

    _service = "projects"
    _action = "get_all"
    _version = "2.20"

    _schema = {
        "definitions": {
            "projects_get_all_response_single": {
                "properties": {
                    "basename": {
                        "description": "Project base name",
                        "type": ["string", "null"],
                    },
                    "company": {
                        "description": "Company id",
                        "type": ["string", "null"],
                    },
                    "created": {
                        "description": "Creation time",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "dataset_stats": {
                        "description": "Project dataset statistics",
                        "properties": {
                            "file_count": {
                                "description": "The number of files stored in the dataset",
                                "type": "integer",
                            },
                            "total_size": {
                                "description": "The total dataset size in bytes",
                                "type": "integer",
                            },
                        },
                        "type": ["object", "null"],
                    },
                    "default_output_destination": {
                        "description": "The default output destination URL for new tasks under this project",
                        "type": ["string", "null"],
                    },
                    "description": {
                        "description": "Project description",
                        "type": ["string", "null"],
                    },
                    "id": {"description": "Project id", "type": ["string", "null"]},
                    "last_update": {
                        "description": "Last update time",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Project name",
                        "type": ["string", "null"],
                    },
                    "own_models": {
                        "description": (
                            "The amount of models under this project (without children projects). "
                            "Returned if 'check_own_contents' flag is set in the request"
                        ),
                        "type": ["integer", "null"],
                    },
                    "own_tasks": {
                        "description": (
                            "The amount of tasks under this project (without children projects). "
                            "Returned if 'check_own_contents' flag is set in the request"
                        ),
                        "type": ["integer", "null"],
                    },
                    "stats": {
                        "description": "Additional project stats",
                        "oneOf": [{"$ref": "#/definitions/stats"}, {"type": "null"}],
                    },
                    "sub_projects": {
                        "description": "The list of sub projects",
                        "items": {
                            "properties": {
                                "id": {
                                    "description": "Subproject ID",
                                    "type": "string",
                                },
                                "name": {
                                    "description": "Subproject name",
                                    "type": "string",
                                },
                            },
                            "type": "object",
                        },
                        "type": ["array", "null"],
                    },
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
            "stats": {
                "properties": {
                    "active": {
                        "description": "Stats for active tasks",
                        "oneOf": [
                            {"$ref": "#/definitions/stats_status_count"},
                            {"type": "null"},
                        ],
                    },
                    "archived": {
                        "description": "Stats for archived tasks",
                        "oneOf": [
                            {"$ref": "#/definitions/stats_status_count"},
                            {"type": "null"},
                        ],
                    },
                },
                "type": "object",
            },
            "stats_status_count": {
                "properties": {
                    "completed_tasks_24h": {
                        "description": "Number of tasks completed in the last 24 hours",
                        "type": ["integer", "null"],
                    },
                    "last_task_run": {
                        "description": "The most recent started time of a task",
                        "type": ["integer", "null"],
                    },
                    "status_count": {
                        "description": "Status counts",
                        "properties": {
                            "closed": {
                                "description": "Number of 'closed' tasks in project",
                                "type": "integer",
                            },
                            "completed": {
                                "description": "Number of 'completed' tasks in project",
                                "type": "integer",
                            },
                            "created": {
                                "description": "Number of 'created' tasks in project",
                                "type": "integer",
                            },
                            "failed": {
                                "description": "Number of 'failed' tasks in project",
                                "type": "integer",
                            },
                            "in_progress": {
                                "description": "Number of 'in_progress' tasks in project",
                                "type": "integer",
                            },
                            "published": {
                                "description": "Number of 'published' tasks in project",
                                "type": "integer",
                            },
                            "queued": {
                                "description": "Number of 'queued' tasks in project",
                                "type": "integer",
                            },
                            "stopped": {
                                "description": "Number of 'stopped' tasks in project",
                                "type": "integer",
                            },
                            "unknown": {
                                "description": "Number of 'unknown' tasks in project",
                                "type": "integer",
                            },
                        },
                        "type": ["object", "null"],
                    },
                    "total_runtime": {
                        "description": "Total run time of all tasks in project (in seconds)",
                        "type": ["integer", "null"],
                    },
                    "total_tasks": {
                        "description": "Number of tasks",
                        "type": ["integer", "null"],
                    },
                },
                "type": "object",
            },
        },
        "properties": {
            "projects": {
                "description": "Projects list",
                "items": {"$ref": "#/definitions/projects_get_all_response_single"},
                "type": ["array", "null"],
            },
            "scroll_id": {
                "description": "Scroll ID that can be used with the next calls to get_all_ex to retrieve more data",
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, projects=None, scroll_id=None, **kwargs):
        super(GetAllResponse, self).__init__(**kwargs)
        self.projects = projects
        self.scroll_id = scroll_id

    @schema_property("projects")
    def projects(self):
        return self._property_projects

    @projects.setter
    def projects(self, value):
        if value is None:
            self._property_projects = None
            return

        self.assert_isinstance(value, "projects", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [ProjectsGetAllResponseSingle.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "projects", ProjectsGetAllResponseSingle, is_array=True)
        self._property_projects = value

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
    :param project: Project id
    :type project: str
    """

    _service = "projects"
    _action = "get_by_id"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {"project": {"description": "Project id", "type": "string"}},
        "required": ["project"],
        "type": "object",
    }

    def __init__(self, project, **kwargs):
        super(GetByIdRequest, self).__init__(**kwargs)
        self.project = project

    @schema_property("project")
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return

        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value


class GetByIdResponse(Response):
    """
    Response of projects.get_by_id endpoint.

    :param project: Project info
    :type project: Project
    """

    _service = "projects"
    _action = "get_by_id"
    _version = "2.20"

    _schema = {
        "definitions": {
            "project": {
                "properties": {
                    "basename": {
                        "description": "Project base name",
                        "type": ["string", "null"],
                    },
                    "company": {
                        "description": "Company id",
                        "type": ["string", "null"],
                    },
                    "created": {
                        "description": "Creation time",
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "default_output_destination": {
                        "description": "The default output destination URL for new tasks under this project",
                        "type": ["string", "null"],
                    },
                    "description": {
                        "description": "Project description",
                        "type": ["string", "null"],
                    },
                    "id": {"description": "Project id", "type": ["string", "null"]},
                    "last_update": {
                        "description": (
                            "Last project update time. Reflects the last time the project metadata was changed or a"
                            " task in this project has changed status"
                        ),
                        "format": "date-time",
                        "type": ["string", "null"],
                    },
                    "name": {
                        "description": "Project name",
                        "type": ["string", "null"],
                    },
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
            }
        },
        "properties": {
            "project": {
                "description": "Project info",
                "oneOf": [{"$ref": "#/definitions/project"}, {"type": "null"}],
            }
        },
        "type": "object",
    }

    def __init__(self, project=None, **kwargs):
        super(GetByIdResponse, self).__init__(**kwargs)
        self.project = project

    @schema_property("project")
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return
        if isinstance(value, dict):
            value = Project.from_dict(value)
        else:
            self.assert_isinstance(value, "project", Project)
        self._property_project = value


class GetHyperParametersRequest(Request):
    """
    Get a list of all hyper parameter sections and names used in tasks within the given project.

    :param project: Project ID
    :type project: str
    :param page: Page number
    :type page: int
    :param page_size: Page size
    :type page_size: int
    :param include_subprojects: If set to 'true' and the project field is set then the result includes
        hyper parameters from the subproject tasks
    :type include_subprojects: bool
    """

    _service = "projects"
    _action = "get_hyper_parameters"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "include_subprojects": {
                "default": True,
                "description": (
                    "If set to 'true' and the project field is set then the result includes hyper parameters from the"
                    " subproject tasks"
                ),
                "type": "boolean",
            },
            "page": {"default": 0, "description": "Page number", "type": "integer"},
            "page_size": {
                "default": 500,
                "description": "Page size",
                "type": "integer",
            },
            "project": {"description": "Project ID", "type": "string"},
        },
        "required": ["project"],
        "type": "object",
    }

    def __init__(self, project, page=0, page_size=500, include_subprojects=True, **kwargs):
        super(GetHyperParametersRequest, self).__init__(**kwargs)
        self.project = project
        self.page = page
        self.page_size = page_size
        self.include_subprojects = include_subprojects

    @schema_property("project")
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return

        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

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

    @schema_property("include_subprojects")
    def include_subprojects(self):
        return self._property_include_subprojects

    @include_subprojects.setter
    def include_subprojects(self, value):
        if value is None:
            self._property_include_subprojects = None
            return

        self.assert_isinstance(value, "include_subprojects", (bool,))
        self._property_include_subprojects = value


class GetHyperParametersResponse(Response):
    """
    Response of projects.get_hyper_parameters endpoint.

    :param parameters: A list of parameter sections and names
    :type parameters: Sequence[dict]
    :param remaining: Remaining results
    :type remaining: int
    :param total: Total number of results
    :type total: int
    """

    _service = "projects"
    _action = "get_hyper_parameters"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "parameters": {
                "description": "A list of parameter sections and names",
                "items": {"type": "object"},
                "type": ["array", "null"],
            },
            "remaining": {
                "description": "Remaining results",
                "type": ["integer", "null"],
            },
            "total": {
                "description": "Total number of results",
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, parameters=None, remaining=None, total=None, **kwargs):
        super(GetHyperParametersResponse, self).__init__(**kwargs)
        self.parameters = parameters
        self.remaining = remaining
        self.total = total

    @schema_property("parameters")
    def parameters(self):
        return self._property_parameters

    @parameters.setter
    def parameters(self, value):
        if value is None:
            self._property_parameters = None
            return

        self.assert_isinstance(value, "parameters", (list, tuple))

        self.assert_isinstance(value, "parameters", (dict,), is_array=True)
        self._property_parameters = value

    @schema_property("remaining")
    def remaining(self):
        return self._property_remaining

    @remaining.setter
    def remaining(self, value):
        if value is None:
            self._property_remaining = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "remaining", six.integer_types)
        self._property_remaining = value

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


class GetHyperparamValuesRequest(Request):
    """
    Get a list of distinct values for the chosen hyperparameter

    :param projects: Project IDs
    :type projects: Sequence[str]
    :param section: Hyperparameter section name
    :type section: str
    :param name: Hyperparameter name
    :type name: str
    :param allow_public: If set to 'true' then collect values from both company and public tasks otherwise company
        tasks only. The default is 'true'
    :type allow_public: bool
    :param include_subprojects: If set to 'true' and the project field is set then the result includes
        hyper parameters values from the subproject tasks
    :type include_subprojects: bool
    """

    _service = "projects"
    _action = "get_hyperparam_values"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "allow_public": {
                "description": (
                    "If set to 'true' then collect values from both company and public tasks otherwise company tasks"
                    " only. The default is 'true'"
                ),
                "type": "boolean",
            },
            "include_subprojects": {
                "default": True,
                "description": (
                    "If set to 'true' and the project field is set then the result includes hyper parameters values"
                    " from the subproject tasks"
                ),
                "type": "boolean",
            },
            "name": {"description": "Hyperparameter name", "type": "string"},
            "projects": {
                "description": "Project IDs",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "section": {"description": "Hyperparameter section name", "type": "string"},
        },
        "required": ["section", "name"],
        "type": "object",
    }

    def __init__(self, section, name, projects=None, allow_public=None, include_subprojects=True, **kwargs):
        super(GetHyperparamValuesRequest, self).__init__(**kwargs)
        self.projects = projects
        self.section = section
        self.name = name
        self.allow_public = allow_public
        self.include_subprojects = include_subprojects

    @schema_property("projects")
    def projects(self):
        return self._property_projects

    @projects.setter
    def projects(self, value):
        if value is None:
            self._property_projects = None
            return

        self.assert_isinstance(value, "projects", (list, tuple))

        self.assert_isinstance(value, "projects", six.string_types, is_array=True)
        self._property_projects = value

    @schema_property("section")
    def section(self):
        return self._property_section

    @section.setter
    def section(self, value):
        if value is None:
            self._property_section = None
            return

        self.assert_isinstance(value, "section", six.string_types)
        self._property_section = value

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

    @schema_property("allow_public")
    def allow_public(self):
        return self._property_allow_public

    @allow_public.setter
    def allow_public(self, value):
        if value is None:
            self._property_allow_public = None
            return

        self.assert_isinstance(value, "allow_public", (bool,))
        self._property_allow_public = value

    @schema_property("include_subprojects")
    def include_subprojects(self):
        return self._property_include_subprojects

    @include_subprojects.setter
    def include_subprojects(self, value):
        if value is None:
            self._property_include_subprojects = None
            return

        self.assert_isinstance(value, "include_subprojects", (bool,))
        self._property_include_subprojects = value


class GetHyperparamValuesResponse(Response):
    """
    Response of projects.get_hyperparam_values endpoint.

    :param total: Total number of distinct parameter values
    :type total: int
    :param values: The list of the unique values for the parameter
    :type values: Sequence[str]
    """

    _service = "projects"
    _action = "get_hyperparam_values"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "total": {
                "description": "Total number of distinct parameter values",
                "type": ["integer", "null"],
            },
            "values": {
                "description": "The list of the unique values for the parameter",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, total=None, values=None, **kwargs):
        super(GetHyperparamValuesResponse, self).__init__(**kwargs)
        self.total = total
        self.values = values

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

    @schema_property("values")
    def values(self):
        return self._property_values

    @values.setter
    def values(self, value):
        if value is None:
            self._property_values = None
            return

        self.assert_isinstance(value, "values", (list, tuple))

        self.assert_isinstance(value, "values", six.string_types, is_array=True)
        self._property_values = value


class GetModelMetadataKeysRequest(Request):
    """
    Get a list of all metadata keys used in models within the given project.

    :param project: Project ID
    :type project: str
    :param include_subprojects: If set to 'true' and the project field is set then the result includes
        metadata keys from the subproject models
    :type include_subprojects: bool
    :param page: Page number
    :type page: int
    :param page_size: Page size
    :type page_size: int
    """

    _service = "projects"
    _action = "get_model_metadata_keys"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "include_subprojects": {
                "default": True,
                "description": (
                    "If set to 'true' and the project field is set then the result includes metadate keys "
                    "from the subproject models"
                ),
                "type": "boolean",
            },
            "page": {"default": 0, "description": "Page number", "type": "integer"},
            "page_size": {
                "default": 500,
                "description": "Page size",
                "type": "integer",
            },
            "project": {"description": "Project ID", "type": "string"},
        },
        "required": ["project"],
        "type": "object",
    }

    def __init__(self, project, include_subprojects=True, page=0, page_size=500, **kwargs):
        super(GetModelMetadataKeysRequest, self).__init__(**kwargs)
        self.project = project
        self.include_subprojects = include_subprojects
        self.page = page
        self.page_size = page_size

    @schema_property("project")
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return

        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property("include_subprojects")
    def include_subprojects(self):
        return self._property_include_subprojects

    @include_subprojects.setter
    def include_subprojects(self, value):
        if value is None:
            self._property_include_subprojects = None
            return

        self.assert_isinstance(value, "include_subprojects", (bool,))
        self._property_include_subprojects = value

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


class GetModelMetadataKeysResponse(Response):
    """
    Response of projects.get_model_metadata_keys endpoint.

    :param keys: A list of model keys
    :type keys: Sequence[str]
    :param remaining: Remaining results
    :type remaining: int
    :param total: Total number of results
    :type total: int
    """

    _service = "projects"
    _action = "get_model_metadata_keys"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "keys": {
                "description": "A list of model keys",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "remaining": {
                "description": "Remaining results",
                "type": ["integer", "null"],
            },
            "total": {
                "description": "Total number of results",
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, keys=None, remaining=None, total=None, **kwargs):
        super(GetModelMetadataKeysResponse, self).__init__(**kwargs)
        self.keys = keys
        self.remaining = remaining
        self.total = total

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

    @schema_property("remaining")
    def remaining(self):
        return self._property_remaining

    @remaining.setter
    def remaining(self, value):
        if value is None:
            self._property_remaining = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "remaining", six.integer_types)
        self._property_remaining = value

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


class GetModelMetadataValuesRequest(Request):
    """
    Get a list of distinct values for the chosen model metadata key

    :param projects: Project IDs
    :type projects: Sequence[str]
    :param key: Metadata key
    :type key: str
    :param allow_public: If set to 'true' then collect values from both company and public models otherwise company
        models only. The default is 'true'
    :type allow_public: bool
    :param include_subprojects: If set to 'true' and the project field is set then the result includes metadata
        values from the subproject models
    :type include_subprojects: bool
    """

    _service = "projects"
    _action = "get_model_metadata_values"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "allow_public": {
                "description": (
                    "If set to 'true' then collect values from both company and public models otherwise company "
                    "models only. The default is 'true'"
                ),
                "type": "boolean",
            },
            "include_subprojects": {
                "default": True,
                "description": (
                    "If set to 'true' and the project field is set then the result includes metadata values from the "
                    "subproject models"
                ),
                "type": "boolean",
            },
            "key": {"description": "Metadata key", "type": "string"},
            "projects": {
                "description": "Project IDs",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "required": ["key"],
        "type": "object",
    }

    def __init__(self, key, projects=None, allow_public=None, include_subprojects=True, **kwargs):
        super(GetModelMetadataValuesRequest, self).__init__(**kwargs)
        self.projects = projects
        self.key = key
        self.allow_public = allow_public
        self.include_subprojects = include_subprojects

    @schema_property("projects")
    def projects(self):
        return self._property_projects

    @projects.setter
    def projects(self, value):
        if value is None:
            self._property_projects = None
            return

        self.assert_isinstance(value, "projects", (list, tuple))

        self.assert_isinstance(value, "projects", six.string_types, is_array=True)
        self._property_projects = value

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

    @schema_property("allow_public")
    def allow_public(self):
        return self._property_allow_public

    @allow_public.setter
    def allow_public(self, value):
        if value is None:
            self._property_allow_public = None
            return

        self.assert_isinstance(value, "allow_public", (bool,))
        self._property_allow_public = value

    @schema_property("include_subprojects")
    def include_subprojects(self):
        return self._property_include_subprojects

    @include_subprojects.setter
    def include_subprojects(self, value):
        if value is None:
            self._property_include_subprojects = None
            return

        self.assert_isinstance(value, "include_subprojects", (bool,))
        self._property_include_subprojects = value


class GetModelMetadataValuesResponse(Response):
    """
    Response of projects.get_model_metadata_values endpoint.

    :param total: Total number of distinct values
    :type total: int
    :param values: The list of the unique values
    :type values: Sequence[str]
    """

    _service = "projects"
    _action = "get_model_metadata_values"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "total": {
                "description": "Total number of distinct values",
                "type": ["integer", "null"],
            },
            "values": {
                "description": "The list of the unique values",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, total=None, values=None, **kwargs):
        super(GetModelMetadataValuesResponse, self).__init__(**kwargs)
        self.total = total
        self.values = values

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

    @schema_property("values")
    def values(self):
        return self._property_values

    @values.setter
    def values(self, value):
        if value is None:
            self._property_values = None
            return

        self.assert_isinstance(value, "values", (list, tuple))

        self.assert_isinstance(value, "values", six.string_types, is_array=True)
        self._property_values = value


class GetModelTagsRequest(Request):
    """
    Get user and system tags used for the models under the specified projects

    :param include_system: If set to 'true' then the list of the system tags is also returned.
        The default value is 'false'
    :type include_system: bool
    :param projects: The list of projects under which the tags are searched. If not passed or empty then all the
        projects are searched
    :type projects: Sequence[str]
    :param filter: Filter on entities to collect tags from
    :type filter: dict
    """

    _service = "projects"
    _action = "get_model_tags"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "filter": {
                "description": "Filter on entities to collect tags from",
                "properties": {
                    "system_tags": {
                        "description": (
                            "The list of system tag values to filter by. Use 'null' value to specify empty system tags."
                            " Use '__Snot' value to specify that the following value should be excluded"
                        ),
                        "items": {"type": "string"},
                        "type": "array",
                    },
                    "tags": {
                        "description": (
                            "The list of tag values to filter by. Use 'null' value to specify empty tags. Use '__Snot'"
                            " value to specify that the following value should be excluded"
                        ),
                        "items": {"type": "string"},
                        "type": "array",
                    },
                },
                "type": ["object", "null"],
            },
            "include_system": {
                "default": False,
                "description": (
                    "If set to 'true' then the list of the system tags is also returned. The default value is 'false'"
                ),
                "type": ["boolean", "null"],
            },
            "projects": {
                "description": (
                    "The list of projects under which the tags are searched. If not passed or empty then all the"
                    " projects are searched"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, include_system=False, projects=None, filter=None, **kwargs):
        super(GetModelTagsRequest, self).__init__(**kwargs)
        self.include_system = include_system
        self.projects = projects
        self.filter = filter

    @schema_property("include_system")
    def include_system(self):
        return self._property_include_system

    @include_system.setter
    def include_system(self, value):
        if value is None:
            self._property_include_system = None
            return

        self.assert_isinstance(value, "include_system", (bool,))
        self._property_include_system = value

    @schema_property("projects")
    def projects(self):
        return self._property_projects

    @projects.setter
    def projects(self, value):
        if value is None:
            self._property_projects = None
            return

        self.assert_isinstance(value, "projects", (list, tuple))

        self.assert_isinstance(value, "projects", six.string_types, is_array=True)
        self._property_projects = value

    @schema_property("filter")
    def filter(self):
        return self._property_filter

    @filter.setter
    def filter(self, value):
        if value is None:
            self._property_filter = None
            return

        self.assert_isinstance(value, "filter", (dict,))
        self._property_filter = value


class GetModelTagsResponse(Response):
    """
    Response of projects.get_model_tags endpoint.

    :param tags: The list of unique tag values
    :type tags: Sequence[str]
    :param system_tags: The list of unique system tag values. Returned only if 'include_system' is set to 'true'
        in the request
    :type system_tags: Sequence[str]
    """

    _service = "projects"
    _action = "get_model_tags"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "system_tags": {
                "description": (
                    "The list of unique system tag values. Returned only if 'include_system' is set to 'true' in the"
                    " request"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "tags": {
                "description": "The list of unique tag values",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, tags=None, system_tags=None, **kwargs):
        super(GetModelTagsResponse, self).__init__(**kwargs)
        self.tags = tags
        self.system_tags = system_tags

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


class GetProjectTagsRequest(Request):
    """
    Get user and system tags used for the specified projects and their children

    :param include_system: If set to 'true' then the list of the system tags is also returned.
        The default value is 'false'
    :type include_system: bool
    :param projects: The list of projects under which the tags are searched. If not passed or empty then all the
        projects are searched
    :type projects: Sequence[str]
    :param filter: Filter on entities to collect tags from
    :type filter: dict
    """

    _service = "projects"
    _action = "get_project_tags"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "filter": {
                "description": "Filter on entities to collect tags from",
                "properties": {
                    "system_tags": {
                        "description": (
                            "The list of system tag values to filter by. Use 'null' value to specify empty "
                            "system tags. Use '__$not' value to specify that the following value should be excluded"
                        ),
                        "items": {"type": "string"},
                        "type": "array",
                    },
                    "tags": {
                        "description": (
                            "The list of tag values to filter by. Use 'null' value to specify empty tags. "
                            "Use '__$not' value to specify that the following value should be excluded"
                        ),
                        "items": {"type": "string"},
                        "type": "array",
                    },
                },
                "type": ["object", "null"],
            },
            "include_system": {
                "default": False,
                "description": (
                    "If set to 'true' then the list of the system tags is also returned. The default value is 'false'"
                ),
                "type": ["boolean", "null"],
            },
            "projects": {
                "description": (
                    "The list of projects under which the tags are searched. If not passed or empty then all the"
                    " projects are searched"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, include_system=False, projects=None, filter=None, **kwargs):
        super(GetProjectTagsRequest, self).__init__(**kwargs)
        self.include_system = include_system
        self.projects = projects
        self.filter = filter

    @schema_property("include_system")
    def include_system(self):
        return self._property_include_system

    @include_system.setter
    def include_system(self, value):
        if value is None:
            self._property_include_system = None
            return

        self.assert_isinstance(value, "include_system", (bool,))
        self._property_include_system = value

    @schema_property("projects")
    def projects(self):
        return self._property_projects

    @projects.setter
    def projects(self, value):
        if value is None:
            self._property_projects = None
            return

        self.assert_isinstance(value, "projects", (list, tuple))

        self.assert_isinstance(value, "projects", six.string_types, is_array=True)
        self._property_projects = value

    @schema_property("filter")
    def filter(self):
        return self._property_filter

    @filter.setter
    def filter(self, value):
        if value is None:
            self._property_filter = None
            return

        self.assert_isinstance(value, "filter", (dict,))
        self._property_filter = value


class GetProjectTagsResponse(Response):
    """
    Response of projects.get_project_tags endpoint.

    :param tags: The list of unique tag values
    :type tags: Sequence[str]
    :param system_tags: The list of unique system tag values. Returned only if 'include_system' is set to 'true'
        in the request
    :type system_tags: Sequence[str]
    """

    _service = "projects"
    _action = "get_project_tags"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "system_tags": {
                "description": (
                    "The list of unique system tag values. Returned only if 'include_system' is set to 'true' in the"
                    " request"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "tags": {
                "description": "The list of unique tag values",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, tags=None, system_tags=None, **kwargs):
        super(GetProjectTagsResponse, self).__init__(**kwargs)
        self.tags = tags
        self.system_tags = system_tags

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


class GetTaskParentsRequest(Request):
    """
    Get unique parent tasks for the tasks in the specified projects

    :param projects: The list of projects which task parents are retrieved. If not passed or empty then all the
        projects are searched
    :type projects: Sequence[str]
    :param tasks_state: Return parents for tasks in the specified state. If Null is provided, parents for all task
        states will be returned.
    :type tasks_state: str
    :param include_subprojects: If set to 'true' and the projects field is not empty then the result includes tasks
        parents from the subproject tasks
    :type include_subprojects: bool
    """

    _service = "projects"
    _action = "get_task_parents"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "include_subprojects": {
                "default": True,
                "description": (
                    "If set to 'true' and the projects field is not empty then the result includes tasks "
                    "parents from the subproject tasks"
                ),
                "type": ["boolean", "null"],
            },
            "projects": {
                "description": (
                    "The list of projects which task parents are retieved. If not passed or empty then all the "
                    "projects are searched"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "tasks_state": {
                "default": "active",
                "description": (
                    "Return parents for tasks in the specified state. If Null is provided, parents for all "
                    "task states will be returned."
                ),
                "enum": ["active", "archived"],
                "type": ["string", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, projects=None, tasks_state="active", include_subprojects=True, **kwargs):
        super(GetTaskParentsRequest, self).__init__(**kwargs)
        self.projects = projects
        self.tasks_state = tasks_state
        self.include_subprojects = include_subprojects

    @schema_property("projects")
    def projects(self):
        return self._property_projects

    @projects.setter
    def projects(self, value):
        if value is None:
            self._property_projects = None
            return

        self.assert_isinstance(value, "projects", (list, tuple))

        self.assert_isinstance(value, "projects", six.string_types, is_array=True)
        self._property_projects = value

    @schema_property("tasks_state")
    def tasks_state(self):
        return self._property_tasks_state

    @tasks_state.setter
    def tasks_state(self, value):
        if value is None:
            self._property_tasks_state = None
            return

        self.assert_isinstance(value, "tasks_state", six.string_types)
        self._property_tasks_state = value

    @schema_property("include_subprojects")
    def include_subprojects(self):
        return self._property_include_subprojects

    @include_subprojects.setter
    def include_subprojects(self, value):
        if value is None:
            self._property_include_subprojects = None
            return

        self.assert_isinstance(value, "include_subprojects", (bool,))
        self._property_include_subprojects = value


class GetTaskParentsResponse(Response):
    """
    Response of projects.get_task_parents endpoint.

    :param parents: The list of unique task parents sorted by their names
    :type parents: Sequence[dict]
    """

    _service = "projects"
    _action = "get_task_parents"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "parents": {
                "description": "The list of unique task parents sorted by their names",
                "items": {
                    "properties": {
                        "id": {
                            "description": "The ID of the parent task",
                            "type": "string",
                        },
                        "name": {
                            "description": "The name of the parent task",
                            "type": "string",
                        },
                        "project": {
                            "id": {
                                "description": "The ID of the parent task project",
                                "type": "string",
                            },
                            "name": {
                                "description": "The name of the parent task project",
                                "type": "string",
                            },
                            "type": "object",
                        },
                    },
                    "type": "object",
                },
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, parents=None, **kwargs):
        super(GetTaskParentsResponse, self).__init__(**kwargs)
        self.parents = parents

    @schema_property("parents")
    def parents(self):
        return self._property_parents

    @parents.setter
    def parents(self, value):
        if value is None:
            self._property_parents = None
            return

        self.assert_isinstance(value, "parents", (list, tuple))

        self.assert_isinstance(value, "parents", (dict,), is_array=True)
        self._property_parents = value


class GetTaskTagsRequest(Request):
    """
    Get user and system tags used for the tasks under the specified projects

    :param include_system: If set to 'true' then the list of the system tags is also returned.
        The default value is 'false'
    :type include_system: bool
    :param projects: The list of projects under which the tags are searched. If not passed or empty then all the
        projects are searched
    :type projects: Sequence[str]
    :param filter: Filter on entities to collect tags from
    :type filter: dict
    """

    _service = "projects"
    _action = "get_task_tags"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "filter": {
                "description": "Filter on entities to collect tags from",
                "properties": {
                    "system_tags": {
                        "description": (
                            "The list of system tag values to filter by. Use 'null' value to specify empty system tags."
                            " Use '__Snot' value to specify that the following value should be excluded"
                        ),
                        "items": {"type": "string"},
                        "type": "array",
                    },
                    "tags": {
                        "description": (
                            "The list of tag values to filter by. Use 'null' value to specify empty tags. Use '__Snot'"
                            " value to specify that the following value should be excluded"
                        ),
                        "items": {"type": "string"},
                        "type": "array",
                    },
                },
                "type": ["object", "null"],
            },
            "include_system": {
                "default": False,
                "description": (
                    "If set to 'true' then the list of the system tags is also returned. The default value is 'false'"
                ),
                "type": ["boolean", "null"],
            },
            "projects": {
                "description": (
                    "The list of projects under which the tags are searched. If not passed or empty then all the"
                    " projects are searched"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, include_system=False, projects=None, filter=None, **kwargs):
        super(GetTaskTagsRequest, self).__init__(**kwargs)
        self.include_system = include_system
        self.projects = projects
        self.filter = filter

    @schema_property("include_system")
    def include_system(self):
        return self._property_include_system

    @include_system.setter
    def include_system(self, value):
        if value is None:
            self._property_include_system = None
            return

        self.assert_isinstance(value, "include_system", (bool,))
        self._property_include_system = value

    @schema_property("projects")
    def projects(self):
        return self._property_projects

    @projects.setter
    def projects(self, value):
        if value is None:
            self._property_projects = None
            return

        self.assert_isinstance(value, "projects", (list, tuple))

        self.assert_isinstance(value, "projects", six.string_types, is_array=True)
        self._property_projects = value

    @schema_property("filter")
    def filter(self):
        return self._property_filter

    @filter.setter
    def filter(self, value):
        if value is None:
            self._property_filter = None
            return

        self.assert_isinstance(value, "filter", (dict,))
        self._property_filter = value


class GetTaskTagsResponse(Response):
    """
    Response of projects.get_task_tags endpoint.

    :param tags: The list of unique tag values
    :type tags: Sequence[str]
    :param system_tags: The list of unique system tag values. Returned only if 'include_system' is set to 'true'
        in the request
    :type system_tags: Sequence[str]
    """

    _service = "projects"
    _action = "get_task_tags"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "system_tags": {
                "description": (
                    "The list of unique system tag values. Returned only if 'include_system' is set to 'true' in the"
                    " request"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
            "tags": {
                "description": "The list of unique tag values",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, tags=None, system_tags=None, **kwargs):
        super(GetTaskTagsResponse, self).__init__(**kwargs)
        self.tags = tags
        self.system_tags = system_tags

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


class GetUniqueMetricVariantsRequest(Request):
    """
    Get all metric/variant pairs reported for tasks in a specific project.
    If no project is specified, metrics/variant paris reported for all tasks will be returned.
    If the project does not exist, an empty list will be returned.

    :param project: Project ID
    :type project: str
    :param include_subprojects: If set to 'true' and the project field is set then the result includes metrics/variants
        from the subproject tasks
    :type include_subprojects: bool
    """

    _service = "projects"
    _action = "get_unique_metric_variants"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "include_subprojects": {
                "default": True,
                "description": (
                    "If set to 'true' and the project field is set then the result includes metrics/variants from the"
                    " subproject tasks"
                ),
                "type": ["boolean", "null"],
            },
            "project": {"description": "Project ID", "type": ["string", "null"]},
        },
        "type": "object",
    }

    def __init__(self, project=None, include_subprojects=True, **kwargs):
        super(GetUniqueMetricVariantsRequest, self).__init__(**kwargs)
        self.project = project
        self.include_subprojects = include_subprojects

    @schema_property("project")
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return

        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property("include_subprojects")
    def include_subprojects(self):
        return self._property_include_subprojects

    @include_subprojects.setter
    def include_subprojects(self, value):
        if value is None:
            self._property_include_subprojects = None
            return

        self.assert_isinstance(value, "include_subprojects", (bool,))
        self._property_include_subprojects = value


class GetUniqueMetricVariantsResponse(Response):
    """
    Response of projects.get_unique_metric_variants endpoint.

    :param metrics: A list of metric variants reported for tasks in this project
    :type metrics: Sequence[MetricVariantResult]
    """

    _service = "projects"
    _action = "get_unique_metric_variants"
    _version = "2.20"

    _schema = {
        "definitions": {
            "metric_variant_result": {
                "properties": {
                    "metric": {
                        "description": "Metric name",
                        "type": ["string", "null"],
                    },
                    "metric_hash": {
                        "description": (
                            "Metric name hash. Used instead of the metric name when categorizing last metrics events in"
                            " task objects."
                        ),
                        "type": ["string", "null"],
                    },
                    "variant": {
                        "description": "Variant name",
                        "type": ["string", "null"],
                    },
                    "variant_hash": {
                        "description": (
                            "Variant name hash. Used instead of the variant name when categorizing last metrics events"
                            " in task objects."
                        ),
                        "type": ["string", "null"],
                    },
                },
                "type": "object",
            }
        },
        "properties": {
            "metrics": {
                "description": "A list of metric variants reported for tasks in this project",
                "items": {"$ref": "#/definitions/metric_variant_result"},
                "type": ["array", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, metrics=None, **kwargs):
        super(GetUniqueMetricVariantsResponse, self).__init__(**kwargs)
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
        if any(isinstance(v, dict) for v in value):
            value = [MetricVariantResult.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "metrics", MetricVariantResult, is_array=True)
        self._property_metrics = value


class MakePrivateRequest(Request):
    """
    Convert public projects to private

    :param ids: Ids of the projects to convert. Only the projects originated by the
        company can be converted
    :type ids: Sequence[str]
    """

    _service = "projects"
    _action = "make_private"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "ids": {
                "description": (
                    "Ids of the projects to convert. Only the projects originated by the company can be converted"
                ),
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, ids=None, **kwargs):
        super(MakePrivateRequest, self).__init__(**kwargs)
        self.ids = ids

    @schema_property("ids")
    def ids(self):
        return self._property_ids

    @ids.setter
    def ids(self, value):
        if value is None:
            self._property_ids = None
            return

        self.assert_isinstance(value, "ids", (list, tuple))

        self.assert_isinstance(value, "ids", six.string_types, is_array=True)
        self._property_ids = value


class MakePrivateResponse(Response):
    """
    Response of projects.make_private endpoint.

    :param updated: Number of projects updated
    :type updated: int
    """

    _service = "projects"
    _action = "make_private"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "updated": {
                "description": "Number of projects updated",
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, updated=None, **kwargs):
        super(MakePrivateResponse, self).__init__(**kwargs)
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


class MakePublicRequest(Request):
    """
    Convert company projects to public

    :param ids: Ids of the projects to convert
    :type ids: Sequence[str]
    """

    _service = "projects"
    _action = "make_public"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "ids": {
                "description": "Ids of the projects to convert",
                "items": {"type": "string"},
                "type": ["array", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, ids=None, **kwargs):
        super(MakePublicRequest, self).__init__(**kwargs)
        self.ids = ids

    @schema_property("ids")
    def ids(self):
        return self._property_ids

    @ids.setter
    def ids(self, value):
        if value is None:
            self._property_ids = None
            return

        self.assert_isinstance(value, "ids", (list, tuple))

        self.assert_isinstance(value, "ids", six.string_types, is_array=True)
        self._property_ids = value


class MakePublicResponse(Response):
    """
    Response of projects.make_public endpoint.

    :param updated: Number of projects updated
    :type updated: int
    """

    _service = "projects"
    _action = "make_public"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "updated": {
                "description": "Number of projects updated",
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, updated=None, **kwargs):
        super(MakePublicResponse, self).__init__(**kwargs)
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


class MergeRequest(Request):
    """
    Moves all the source project's contents to the destination project and remove the source project

    :param project: Project id
    :type project: str
    :param destination_project: The ID of the destination project
    :type destination_project: str
    """

    _service = "projects"
    _action = "merge"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "destination_project": {
                "description": "The ID of the destination project",
                "type": "string",
            },
            "project": {"description": "Project id", "type": "string"},
        },
        "required": ["project"],
        "type": "object",
    }

    def __init__(self, project, destination_project=None, **kwargs):
        super(MergeRequest, self).__init__(**kwargs)
        self.project = project
        self.destination_project = destination_project

    @schema_property("project")
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return

        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property("destination_project")
    def destination_project(self):
        return self._property_destination_project

    @destination_project.setter
    def destination_project(self, value):
        if value is None:
            self._property_destination_project = None
            return

        self.assert_isinstance(value, "destination_project", six.string_types)
        self._property_destination_project = value


class MergeResponse(Response):
    """
    Response of projects.merge endpoint.

    :param moved_entities: The number of tasks and models moved from the merged
        project into the destination
    :type moved_entities: int
    :param moved_projects: The number of child projects moved from the merged
        project into the destination
    :type moved_projects: int
    """

    _service = "projects"
    _action = "merge"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "moved_entities": {
                "description": (
                    "The number of tasks, models and dataviews moved from the merged project into the destination"
                ),
                "type": ["integer", "null"],
            },
            "moved_projects": {
                "description": "The number of child projects moved from the merged project into the destination",
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, moved_entities=None, moved_projects=None, **kwargs):
        super(MergeResponse, self).__init__(**kwargs)
        self.moved_entities = moved_entities
        self.moved_projects = moved_projects

    @schema_property("moved_entities")
    def moved_entities(self):
        return self._property_moved_entities

    @moved_entities.setter
    def moved_entities(self, value):
        if value is None:
            self._property_moved_entities = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "moved_entities", six.integer_types)
        self._property_moved_entities = value

    @schema_property("moved_projects")
    def moved_projects(self):
        return self._property_moved_projects

    @moved_projects.setter
    def moved_projects(self, value):
        if value is None:
            self._property_moved_projects = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "moved_projects", six.integer_types)
        self._property_moved_projects = value


class MoveRequest(Request):
    """
    Moves a project and all of its subprojects under the different location

    :param project: Project id
    :type project: str
    :param new_location: The name location for the project
    :type new_location: str
    """

    _service = "projects"
    _action = "move"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "new_location": {
                "description": "The name location for the project",
                "type": "string",
            },
            "project": {"description": "Project id", "type": "string"},
        },
        "required": ["project"],
        "type": "object",
    }

    def __init__(self, project, new_location=None, **kwargs):
        super(MoveRequest, self).__init__(**kwargs)
        self.project = project
        self.new_location = new_location

    @schema_property("project")
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return

        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property("new_location")
    def new_location(self):
        return self._property_new_location

    @new_location.setter
    def new_location(self, value):
        if value is None:
            self._property_new_location = None
            return

        self.assert_isinstance(value, "new_location", six.string_types)
        self._property_new_location = value


class MoveResponse(Response):
    """
    Response of projects.move endpoint.

    :param moved: The number of projects moved
    :type moved: int
    """

    _service = "projects"
    _action = "move"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "moved": {
                "description": "The number of projects moved",
                "type": ["integer", "null"],
            }
        },
        "type": "object",
    }

    def __init__(self, moved=None, **kwargs):
        super(MoveResponse, self).__init__(**kwargs)
        self.moved = moved

    @schema_property("moved")
    def moved(self):
        return self._property_moved

    @moved.setter
    def moved(self, value):
        if value is None:
            self._property_moved = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "moved", six.integer_types)
        self._property_moved = value


class UpdateRequest(Request):
    """
    Update project information

    :param project: Project id
    :type project: str
    :param name: Project name. Unique within the company.
    :type name: str
    :param description: Project description
    :type description: str
    :param tags: User-defined tags list
    :type tags: Sequence[str]
    :param system_tags: System tags list. This field is reserved for system use, please don't use it.
    :type system_tags: Sequence[str]
    :param default_output_destination: The default output destination URL for new tasks under this project
    :type default_output_destination: str
    """

    _service = "projects"
    _action = "update"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {
            "default_output_destination": {
                "description": "The default output destination URL for new tasks under this project",
                "type": "string",
            },
            "description": {"description": "Project description", "type": "string"},
            "name": {
                "description": "Project name. Unique within the company.",
                "type": "string",
            },
            "project": {"description": "Project id", "type": "string"},
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
        "required": ["project"],
        "type": "object",
    }

    def __init__(
        self,
        project,
        name=None,
        description=None,
        tags=None,
        system_tags=None,
        default_output_destination=None,
        **kwargs
    ):
        super(UpdateRequest, self).__init__(**kwargs)
        self.project = project
        self.name = name
        self.description = description
        self.tags = tags
        self.system_tags = system_tags
        self.default_output_destination = default_output_destination

    @schema_property("project")
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return

        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

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

    @schema_property("description")
    def description(self):
        return self._property_description

    @description.setter
    def description(self, value):
        if value is None:
            self._property_description = None
            return

        self.assert_isinstance(value, "description", six.string_types)
        self._property_description = value

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

    @schema_property("default_output_destination")
    def default_output_destination(self):
        return self._property_default_output_destination

    @default_output_destination.setter
    def default_output_destination(self, value):
        if value is None:
            self._property_default_output_destination = None
            return

        self.assert_isinstance(value, "default_output_destination", six.string_types)
        self._property_default_output_destination = value


class UpdateResponse(Response):
    """
    Response of projects.update endpoint.

    :param updated: Number of projects updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """

    _service = "projects"
    _action = "update"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "fields": {
                "additionalProperties": True,
                "description": "Updated fields names and values",
                "type": ["object", "null"],
            },
            "updated": {
                "description": "Number of projects updated (0 or 1)",
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


class ValidateDeleteRequest(Request):
    """
    Validates that the project exists and can be deleted

    :param project: Project ID
    :type project: str
    """

    _service = "projects"
    _action = "validate_delete"
    _version = "2.20"
    _schema = {
        "definitions": {},
        "properties": {"project": {"description": "Project ID", "type": "string"}},
        "required": ["project"],
        "type": "object",
    }

    def __init__(self, project, **kwargs):
        super(ValidateDeleteRequest, self).__init__(**kwargs)
        self.project = project

    @schema_property("project")
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return

        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value


class ValidateDeleteResponse(Response):
    """
    Response of projects.validate_delete endpoint.

    :param tasks: The total number of tasks under the project and all its children
    :type tasks: int
    :param non_archived_tasks: The total number of non-archived tasks under the project and all its children
    :type non_archived_tasks: int
    :param models: The total number of models under the project and all its children
    :type models: int
    :param non_archived_models: The total number of non-archived models under the project and all its children
    :type non_archived_models: int
    """

    _service = "projects"
    _action = "validate_delete"
    _version = "2.20"

    _schema = {
        "definitions": {},
        "properties": {
            "models": {
                "description": "The total number of models under the project and all its children",
                "type": ["integer", "null"],
            },
            "non_archived_models": {
                "description": "The total number of non-archived models under the project and all its children",
                "type": ["integer", "null"],
            },
            "non_archived_tasks": {
                "description": "The total number of non-archived tasks under the project and all its children",
                "type": ["integer", "null"],
            },
            "tasks": {
                "description": "The total number of tasks under the project and all its children",
                "type": ["integer", "null"],
            },
        },
        "type": "object",
    }

    def __init__(self, tasks=None, non_archived_tasks=None, models=None, non_archived_models=None, **kwargs):
        super(ValidateDeleteResponse, self).__init__(**kwargs)
        self.tasks = tasks
        self.non_archived_tasks = non_archived_tasks
        self.models = models
        self.non_archived_models = non_archived_models

    @schema_property("tasks")
    def tasks(self):
        return self._property_tasks

    @tasks.setter
    def tasks(self, value):
        if value is None:
            self._property_tasks = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "tasks", six.integer_types)
        self._property_tasks = value

    @schema_property("non_archived_tasks")
    def non_archived_tasks(self):
        return self._property_non_archived_tasks

    @non_archived_tasks.setter
    def non_archived_tasks(self, value):
        if value is None:
            self._property_non_archived_tasks = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "non_archived_tasks", six.integer_types)
        self._property_non_archived_tasks = value

    @schema_property("models")
    def models(self):
        return self._property_models

    @models.setter
    def models(self, value):
        if value is None:
            self._property_models = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "models", six.integer_types)
        self._property_models = value

    @schema_property("non_archived_models")
    def non_archived_models(self):
        return self._property_non_archived_models

    @non_archived_models.setter
    def non_archived_models(self, value):
        if value is None:
            self._property_non_archived_models = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "non_archived_models", six.integer_types)
        self._property_non_archived_models = value


response_mapping = {
    CreateRequest: CreateResponse,
    GetByIdRequest: GetByIdResponse,
    GetAllRequest: GetAllResponse,
    UpdateRequest: UpdateResponse,
    MoveRequest: MoveResponse,
    MergeRequest: MergeResponse,
    ValidateDeleteRequest: ValidateDeleteResponse,
    DeleteRequest: DeleteResponse,
    GetUniqueMetricVariantsRequest: GetUniqueMetricVariantsResponse,
    GetHyperparamValuesRequest: GetHyperparamValuesResponse,
    GetHyperParametersRequest: GetHyperParametersResponse,
    GetModelMetadataValuesRequest: GetModelMetadataValuesResponse,
    GetModelMetadataKeysRequest: GetModelMetadataKeysResponse,
    GetProjectTagsRequest: GetProjectTagsResponse,
    GetTaskTagsRequest: GetTaskTagsResponse,
    GetModelTagsRequest: GetModelTagsResponse,
    MakePublicRequest: MakePublicResponse,
    MakePrivateRequest: MakePrivateResponse,
    GetTaskParentsRequest: GetTaskParentsResponse,
}
