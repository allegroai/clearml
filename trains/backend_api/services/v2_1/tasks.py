"""
tasks service

Provides a management API for tasks in the system.
"""
import enum
from datetime import datetime

import six
from dateutil.parser import parse as parse_datetime

from ....backend_api.session import Request, BatchRequest, Response, NonStrictDataModel, schema_property, StringEnum


class FilterByRoiEnum(StringEnum):
    disabled = "disabled"
    no_rois = "no_rois"
    label_rules = "label_rules"


class FilterLabelRule(NonStrictDataModel):
    """
    :param label: Lucene format query (see lucene query syntax). Default search
        field is label.keyword and default operator is AND, so searching for:
        'Bus Stop' Blue
        is equivalent to:
        Label.keyword:'Bus Stop' AND label.keyword:'Blue'
    :type label: str
    :param count_range: Range of times ROI appears in the frame (min, max). -1 for
        not applicable. Both integers must be larger than or equal to -1. 2nd integer
        (max) must be either -1 or larger than or equal to the 1st integer (min)
    :type count_range: Sequence[int]
    :param conf_range: Range of ROI confidence level in the frame (min, max). -1
        for not applicable Both min and max can be either -1 or positive. 2nd number
        (max) must be either -1 or larger than or equal to the 1st number (min)
    :type conf_range: Sequence[float]
    """
    _schema = {
        'properties': {
            'conf_range': {
                'description': 'Range of ROI confidence level in the frame (min, max). -1 for not applicable\n            Both min and max can be either -1 or positive.\n            2nd number (max) must be either -1 or larger than or equal to the 1st number (min)',
                'items': {'type': 'number'},
                'maxItems': 2,
                'minItems': 1,
                'type': 'array',
            },
            'count_range': {
                'description': 'Range of times ROI appears in the frame (min, max). -1 for not applicable.\n            Both integers must be larger than or equal to -1.\n            2nd integer (max) must be either -1 or larger than or equal to the 1st integer (min)',
                'items': {'type': 'integer'},
                'maxItems': 2,
                'minItems': 1,
                'type': 'array',
            },
            'label': {
                'description': "Lucene format query (see lucene query syntax).\nDefault search field is label.keyword and default operator is AND, so searching for:\n\n'Bus Stop' Blue\n\nis equivalent to:\n\nLabel.keyword:'Bus Stop' AND label.keyword:'Blue'",
                'type': 'string',
            },
        },
        'required': ['label'],
        'type': 'object',
    }
    def __init__(
            self, label, count_range=None, conf_range=None, **kwargs):
        super(FilterLabelRule, self).__init__(**kwargs)
        self.label = label
        self.count_range = count_range
        self.conf_range = conf_range

    @schema_property('label')
    def label(self):
        return self._property_label

    @label.setter
    def label(self, value):
        if value is None:
            self._property_label = None
            return
        
        self.assert_isinstance(value, "label", six.string_types)
        self._property_label = value

    @schema_property('count_range')
    def count_range(self):
        return self._property_count_range

    @count_range.setter
    def count_range(self, value):
        if value is None:
            self._property_count_range = None
            return
        
        self.assert_isinstance(value, "count_range", (list, tuple))
        value = [int(v) if isinstance(v, float) and v.is_integer() else v for v in value]

        self.assert_isinstance(value, "count_range", six.integer_types, is_array=True)
        self._property_count_range = value

    @schema_property('conf_range')
    def conf_range(self):
        return self._property_conf_range

    @conf_range.setter
    def conf_range(self, value):
        if value is None:
            self._property_conf_range = None
            return
        
        self.assert_isinstance(value, "conf_range", (list, tuple))
        
        self.assert_isinstance(value, "conf_range", six.integer_types + (float,), is_array=True)
        self._property_conf_range = value


class FilterRule(NonStrictDataModel):
    """
    :param label_rules: List of FilterLabelRule ('AND' connection)
        disabled - No filtering by ROIs. Select all frames, even if they don't have
        ROIs (all frames)
        no_rois - Select only frames without ROIs (empty frames)
        label_rules - Select frames according to label rules
    :type label_rules: Sequence[FilterLabelRule]
    :param filter_by_roi: Type of filter
    :type filter_by_roi: FilterByRoiEnum
    :param frame_query: Frame filter, in Lucene query syntax
    :type frame_query: str
    :param sources_query: Sources filter, in Lucene query syntax. Filters sources
        in each frame.
    :type sources_query: str
    :param dataset: Dataset ID. Must be a dataset which is in the task's view. If
        set to '*' all datasets in View are used.
    :type dataset: str
    :param version: Dataset version to apply rule to. Must belong to the dataset
        and be in the task's view. If set to '*' all version of the datasets in View
        are used.
    :type version: str
    :param weight: Rule weight. Default is 1
    :type weight: float
    """
    _schema = {
        'properties': {
            'dataset': {
                'description': "Dataset ID. Must be a dataset which is in the task's view. If set to '*' all datasets in View are used.",
                'type': 'string',
            },
            'filter_by_roi': {
                '$ref': '#/definitions/filter_by_roi_enum',
                'description': 'Type of filter',
            },
            'frame_query': {
                'description': 'Frame filter, in Lucene query syntax',
                'type': 'string',
            },
            'label_rules': {
                'description': "List of FilterLabelRule ('AND' connection)\n\ndisabled - No filtering by ROIs. Select all frames, even if they don't have ROIs (all frames)\n\nno_rois - Select only frames without ROIs (empty frames)\n\nlabel_rules - Select frames according to label rules",
                'items': {'$ref': '#/definitions/filter_label_rule'},
                'type': ['array', 'null'],
            },
            'sources_query': {
                'description': 'Sources filter, in Lucene query syntax. Filters sources in each frame.',
                'type': 'string',
            },
            'version': {
                'description': "Dataset version to apply rule to. Must belong to the dataset and be in the task's view. If set to '*' all version of the datasets in View are used.",
                'type': 'string',
            },
            'weight': {
                'description': 'Rule weight. Default is 1',
                'type': 'number',
            },
        },
        'required': ['filter_by_roi'],
        'type': 'object',
    }
    def __init__(
            self, filter_by_roi, label_rules=None, frame_query=None, sources_query=None, dataset=None, version=None, weight=None, **kwargs):
        super(FilterRule, self).__init__(**kwargs)
        self.label_rules = label_rules
        self.filter_by_roi = filter_by_roi
        self.frame_query = frame_query
        self.sources_query = sources_query
        self.dataset = dataset
        self.version = version
        self.weight = weight

    @schema_property('label_rules')
    def label_rules(self):
        return self._property_label_rules

    @label_rules.setter
    def label_rules(self, value):
        if value is None:
            self._property_label_rules = None
            return
        
        self.assert_isinstance(value, "label_rules", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [FilterLabelRule.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "label_rules", FilterLabelRule, is_array=True)
        self._property_label_rules = value

    @schema_property('filter_by_roi')
    def filter_by_roi(self):
        return self._property_filter_by_roi

    @filter_by_roi.setter
    def filter_by_roi(self, value):
        if value is None:
            self._property_filter_by_roi = None
            return
        if isinstance(value, six.string_types):
            try:
                value = FilterByRoiEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "filter_by_roi", enum.Enum)
        self._property_filter_by_roi = value

    @schema_property('frame_query')
    def frame_query(self):
        return self._property_frame_query

    @frame_query.setter
    def frame_query(self, value):
        if value is None:
            self._property_frame_query = None
            return
        
        self.assert_isinstance(value, "frame_query", six.string_types)
        self._property_frame_query = value

    @schema_property('sources_query')
    def sources_query(self):
        return self._property_sources_query

    @sources_query.setter
    def sources_query(self, value):
        if value is None:
            self._property_sources_query = None
            return
        
        self.assert_isinstance(value, "sources_query", six.string_types)
        self._property_sources_query = value

    @schema_property('dataset')
    def dataset(self):
        return self._property_dataset

    @dataset.setter
    def dataset(self, value):
        if value is None:
            self._property_dataset = None
            return
        
        self.assert_isinstance(value, "dataset", six.string_types)
        self._property_dataset = value

    @schema_property('version')
    def version(self):
        return self._property_version

    @version.setter
    def version(self, value):
        if value is None:
            self._property_version = None
            return
        
        self.assert_isinstance(value, "version", six.string_types)
        self._property_version = value

    @schema_property('weight')
    def weight(self):
        return self._property_weight

    @weight.setter
    def weight(self, value):
        if value is None:
            self._property_weight = None
            return
        
        self.assert_isinstance(value, "weight", six.integer_types + (float,))
        self._property_weight = value


class MultiFieldPatternData(NonStrictDataModel):
    """
    :param pattern: Pattern string (regex)
    :type pattern: str
    :param fields: List of field names
    :type fields: Sequence[str]
    """
    _schema = {
        'properties': {
            'fields': {
                'description': 'List of field names',
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
            'pattern': {
                'description': 'Pattern string (regex)',
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, pattern=None, fields=None, **kwargs):
        super(MultiFieldPatternData, self).__init__(**kwargs)
        self.pattern = pattern
        self.fields = fields

    @schema_property('pattern')
    def pattern(self):
        return self._property_pattern

    @pattern.setter
    def pattern(self, value):
        if value is None:
            self._property_pattern = None
            return
        
        self.assert_isinstance(value, "pattern", six.string_types)
        self._property_pattern = value

    @schema_property('fields')
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


class Script(NonStrictDataModel):
    """
    :param binary: Binary to use when running the script
    :type binary: str
    :param repository: Name of the repository where the script is located
    :type repository: str
    :param tag: Repository tag
    :type tag: str
    :param branch: Repository branch id If not provided and tag not provided,
        default repository branch is used.
    :type branch: str
    :param version_num: Version (changeset) number. Optional (default is head
        version) Unused if tag is provided.
    :type version_num: str
    :param entry_point: Path to execute within the repository
    :type entry_point: str
    :param working_dir: Path to the folder from which to run the script Default -
        root folder of repository[f]
    :type working_dir: str
    :param requirements: A JSON object containing requirements strings by key
    :type requirements: dict
    """
    _schema = {
        'properties': {
            'binary': {
                'default': 'python',
                'description': 'Binary to use when running the script',
                'type': ['string', 'null'],
            },
            'branch': {
                'description': 'Repository branch id If not provided and tag not provided, default repository branch is used.',
                'type': ['string', 'null'],
            },
            'entry_point': {
                'description': 'Path to execute within the repository',
                'type': ['string', 'null'],
            },
            'repository': {
                'description': 'Name of the repository where the script is located',
                'type': ['string', 'null'],
            },
            'requirements': {
                'description': 'A JSON object containing requirements strings by key',
                'type': ['object', 'null'],
            },
            'tag': {'description': 'Repository tag', 'type': ['string', 'null']},
            'version_num': {
                'description': 'Version (changeset) number. Optional (default is head version) Unused if tag is provided.',
                'type': ['string', 'null'],
            },
            'working_dir': {
                'description': 'Path to the folder from which to run the script Default - root folder of repository[f]',
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, binary="python", repository=None, tag=None, branch=None, version_num=None, entry_point=None, working_dir=None, requirements=None, **kwargs):
        super(Script, self).__init__(**kwargs)
        self.binary = binary
        self.repository = repository
        self.tag = tag
        self.branch = branch
        self.version_num = version_num
        self.entry_point = entry_point
        self.working_dir = working_dir
        self.requirements = requirements

    @schema_property('binary')
    def binary(self):
        return self._property_binary

    @binary.setter
    def binary(self, value):
        if value is None:
            self._property_binary = None
            return
        
        self.assert_isinstance(value, "binary", six.string_types)
        self._property_binary = value

    @schema_property('repository')
    def repository(self):
        return self._property_repository

    @repository.setter
    def repository(self, value):
        if value is None:
            self._property_repository = None
            return
        
        self.assert_isinstance(value, "repository", six.string_types)
        self._property_repository = value

    @schema_property('tag')
    def tag(self):
        return self._property_tag

    @tag.setter
    def tag(self, value):
        if value is None:
            self._property_tag = None
            return
        
        self.assert_isinstance(value, "tag", six.string_types)
        self._property_tag = value

    @schema_property('branch')
    def branch(self):
        return self._property_branch

    @branch.setter
    def branch(self, value):
        if value is None:
            self._property_branch = None
            return
        
        self.assert_isinstance(value, "branch", six.string_types)
        self._property_branch = value

    @schema_property('version_num')
    def version_num(self):
        return self._property_version_num

    @version_num.setter
    def version_num(self, value):
        if value is None:
            self._property_version_num = None
            return
        
        self.assert_isinstance(value, "version_num", six.string_types)
        self._property_version_num = value

    @schema_property('entry_point')
    def entry_point(self):
        return self._property_entry_point

    @entry_point.setter
    def entry_point(self, value):
        if value is None:
            self._property_entry_point = None
            return
        
        self.assert_isinstance(value, "entry_point", six.string_types)
        self._property_entry_point = value

    @schema_property('working_dir')
    def working_dir(self):
        return self._property_working_dir

    @working_dir.setter
    def working_dir(self, value):
        if value is None:
            self._property_working_dir = None
            return
        
        self.assert_isinstance(value, "working_dir", six.string_types)
        self._property_working_dir = value

    @schema_property('requirements')
    def requirements(self):
        return self._property_requirements

    @requirements.setter
    def requirements(self, value):
        if value is None:
            self._property_requirements = None
            return
        
        self.assert_isinstance(value, "requirements", (dict,))
        self._property_requirements = value


class LabelSource(NonStrictDataModel):
    """
    :param labels: List of source labels (AND connection). '*' indicates any label.
        Labels must exist in at least one of the dataset versions in the task's view
    :type labels: Sequence[str]
    :param dataset: Source dataset id. '*' for all datasets in view
    :type dataset: str
    :param version: Source dataset version id. Default is '*' (for all versions in
        dataset in the view) Version must belong to the selected dataset, and must be
        in the task's view[i]
    :type version: str
    """
    _schema = {
        'properties': {
            'dataset': {
                'description': "Source dataset id. '*' for all datasets in view",
                'type': ['string', 'null'],
            },
            'labels': {
                'description': "List of source labels (AND connection). '*' indicates any label. Labels must exist in at least one of the dataset versions in the task's view",
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
            'version': {
                'description': "Source dataset version id. Default is '*' (for all versions in dataset in the view) Version must belong to the selected dataset, and must be in the task's view[i]",
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, labels=None, dataset=None, version=None, **kwargs):
        super(LabelSource, self).__init__(**kwargs)
        self.labels = labels
        self.dataset = dataset
        self.version = version

    @schema_property('labels')
    def labels(self):
        return self._property_labels

    @labels.setter
    def labels(self, value):
        if value is None:
            self._property_labels = None
            return
        
        self.assert_isinstance(value, "labels", (list, tuple))
        
        self.assert_isinstance(value, "labels", six.string_types, is_array=True)
        self._property_labels = value

    @schema_property('dataset')
    def dataset(self):
        return self._property_dataset

    @dataset.setter
    def dataset(self, value):
        if value is None:
            self._property_dataset = None
            return
        
        self.assert_isinstance(value, "dataset", six.string_types)
        self._property_dataset = value

    @schema_property('version')
    def version(self):
        return self._property_version

    @version.setter
    def version(self, value):
        if value is None:
            self._property_version = None
            return
        
        self.assert_isinstance(value, "version", six.string_types)
        self._property_version = value


class MappingRule(NonStrictDataModel):
    """
    :param source: Source label info
    :type source: LabelSource
    :param target: Target label name
    :type target: str
    """
    _schema = {
        'properties': {
            'source': {
                'description': 'Source label info',
                'oneOf': [{'$ref': '#/definitions/label_source'}, {'type': 'null'}],
            },
            'target': {
                'description': 'Target label name',
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, source=None, target=None, **kwargs):
        super(MappingRule, self).__init__(**kwargs)
        self.source = source
        self.target = target

    @schema_property('source')
    def source(self):
        return self._property_source

    @source.setter
    def source(self, value):
        if value is None:
            self._property_source = None
            return
        if isinstance(value, dict):
            value = LabelSource.from_dict(value)
        else:
            self.assert_isinstance(value, "source", LabelSource)
        self._property_source = value

    @schema_property('target')
    def target(self):
        return self._property_target

    @target.setter
    def target(self, value):
        if value is None:
            self._property_target = None
            return
        
        self.assert_isinstance(value, "target", six.string_types)
        self._property_target = value


class Mapping(NonStrictDataModel):
    """
    :param rules: Rules list
    :type rules: Sequence[MappingRule]
    """
    _schema = {
        'properties': {
            'rules': {
                'description': 'Rules list',
                'items': {'$ref': '#/definitions/mapping_rule'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, rules=None, **kwargs):
        super(Mapping, self).__init__(**kwargs)
        self.rules = rules

    @schema_property('rules')
    def rules(self):
        return self._property_rules

    @rules.setter
    def rules(self, value):
        if value is None:
            self._property_rules = None
            return
        
        self.assert_isinstance(value, "rules", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [MappingRule.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "rules", MappingRule, is_array=True)
        self._property_rules = value


class Filtering(NonStrictDataModel):
    """
    :param filtering_rules: List of FilterRule ('OR' connection)
    :type filtering_rules: Sequence[FilterRule]
    :param output_rois: 'all_in_frame' - all rois for a frame are returned
        'only_filtered' - only rois which led this frame to be selected
        'frame_per_roi' - single roi per frame. Frame can be returned multiple times
        with a different roi each time.
        Note: this should be used for Training tasks only
        Note: frame_per_roi implies that only filtered rois will be returned
    :type output_rois: OutputRoisEnum
    """
    _schema = {
        'properties': {
            'filtering_rules': {
                'description': "List of FilterRule ('OR' connection)",
                'items': {'$ref': '#/definitions/filter_rule'},
                'type': ['array', 'null'],
            },
            'output_rois': {
                'description': "'all_in_frame' - all rois for a frame are returned\n\n'only_filtered' - only rois which led this frame to be selected\n\n'frame_per_roi' - single roi per frame. Frame can be returned multiple times with a different roi each time.\n\nNote: this should be used for Training tasks only\n\nNote: frame_per_roi implies that only filtered rois will be returned\n                ",
                'oneOf': [
                    {'$ref': '#/definitions/output_rois_enum'},
                    {'type': 'null'},
                ],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, filtering_rules=None, output_rois=None, **kwargs):
        super(Filtering, self).__init__(**kwargs)
        self.filtering_rules = filtering_rules
        self.output_rois = output_rois

    @schema_property('filtering_rules')
    def filtering_rules(self):
        return self._property_filtering_rules

    @filtering_rules.setter
    def filtering_rules(self, value):
        if value is None:
            self._property_filtering_rules = None
            return
        
        self.assert_isinstance(value, "filtering_rules", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [FilterRule.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "filtering_rules", FilterRule, is_array=True)
        self._property_filtering_rules = value

    @schema_property('output_rois')
    def output_rois(self):
        return self._property_output_rois

    @output_rois.setter
    def output_rois(self, value):
        if value is None:
            self._property_output_rois = None
            return
        if isinstance(value, six.string_types):
            try:
                value = OutputRoisEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "output_rois", enum.Enum)
        self._property_output_rois = value


class Jump(NonStrictDataModel):
    """
    :param time: Max time in milliseconds between frames
    :type time: int
    """
    _schema = {
        'properties': {
            'time': {
                'description': 'Max time in milliseconds between frames',
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, time=None, **kwargs):
        super(Jump, self).__init__(**kwargs)
        self.time = time

    @schema_property('time')
    def time(self):
        return self._property_time

    @time.setter
    def time(self, value):
        if value is None:
            self._property_time = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "time", six.integer_types)
        self._property_time = value


class AugmentationSet(NonStrictDataModel):
    """
    :param cls: Augmentation class
    :type cls: str
    :param types: Augmentation type
    :type types: Sequence[str]
    :param strength: Augmentation strength. Range [0,).
    :type strength: float
    :param arguments: Arguments dictionary per custom augmentation type.
    :type arguments: dict
    """
    _schema = {
        'properties': {
            'arguments': {
                'additionalProperties': {
                    'additionalProperties': True,
                    'type': 'object',
                },
                'description': 'Arguments dictionary per custom augmentation type.',
                'type': ['object', 'null'],
            },
            'cls': {
                'description': 'Augmentation class',
                'type': ['string', 'null'],
            },
            'strength': {
                'description': 'Augmentation strength. Range [0,).',
                'minimum': 0,
                'type': ['number', 'null'],
            },
            'types': {
                'description': 'Augmentation type',
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, cls=None, types=None, strength=None, arguments=None, **kwargs):
        super(AugmentationSet, self).__init__(**kwargs)
        self.cls = cls
        self.types = types
        self.strength = strength
        self.arguments = arguments

    @schema_property('cls')
    def cls(self):
        return self._property_cls

    @cls.setter
    def cls(self, value):
        if value is None:
            self._property_cls = None
            return
        
        self.assert_isinstance(value, "cls", six.string_types)
        self._property_cls = value

    @schema_property('types')
    def types(self):
        return self._property_types

    @types.setter
    def types(self, value):
        if value is None:
            self._property_types = None
            return
        
        self.assert_isinstance(value, "types", (list, tuple))
        
        self.assert_isinstance(value, "types", six.string_types, is_array=True)
        self._property_types = value

    @schema_property('strength')
    def strength(self):
        return self._property_strength

    @strength.setter
    def strength(self, value):
        if value is None:
            self._property_strength = None
            return
        
        self.assert_isinstance(value, "strength", six.integer_types + (float,))
        self._property_strength = value

    @schema_property('arguments')
    def arguments(self):
        return self._property_arguments

    @arguments.setter
    def arguments(self, value):
        if value is None:
            self._property_arguments = None
            return
        
        self.assert_isinstance(value, "arguments", (dict,))
        self._property_arguments = value


class Augmentation(NonStrictDataModel):
    """
    :param sets: List of augmentation sets
    :type sets: Sequence[AugmentationSet]
    :param crop_around_rois: Crop image data around all frame ROIs
    :type crop_around_rois: bool
    """
    _schema = {
        'properties': {
            'crop_around_rois': {
                'description': 'Crop image data around all frame ROIs',
                'type': ['boolean', 'null'],
            },
            'sets': {
                'description': 'List of augmentation sets',
                'items': {'$ref': '#/definitions/augmentation_set'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, sets=None, crop_around_rois=None, **kwargs):
        super(Augmentation, self).__init__(**kwargs)
        self.sets = sets
        self.crop_around_rois = crop_around_rois

    @schema_property('sets')
    def sets(self):
        return self._property_sets

    @sets.setter
    def sets(self, value):
        if value is None:
            self._property_sets = None
            return
        
        self.assert_isinstance(value, "sets", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [AugmentationSet.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "sets", AugmentationSet, is_array=True)
        self._property_sets = value

    @schema_property('crop_around_rois')
    def crop_around_rois(self):
        return self._property_crop_around_rois

    @crop_around_rois.setter
    def crop_around_rois(self, value):
        if value is None:
            self._property_crop_around_rois = None
            return
        
        self.assert_isinstance(value, "crop_around_rois", (bool,))
        self._property_crop_around_rois = value


class Iteration(NonStrictDataModel):
    """
    Sequential Iteration API configuration

    :param order: Input frames order. Values: 'sequential', 'random' In Sequential
        mode frames will be returned according to the order in which the frames were
        added to the dataset.
    :type order: str
    :param jump: Jump entry
    :type jump: Jump
    :param min_sequence: Length (in ms) of video clips to return. This is used in
        random order, and in sequential order only if jumping is provided and only for
        video frames
    :type min_sequence: int
    :param infinite: Infinite iteration
    :type infinite: bool
    :param limit: Maximum frames per task. If not passed, frames will end when no
        more matching frames are found, unless infinite is True.
    :type limit: int
    :param random_seed: Random seed used during iteration
    :type random_seed: int
    """
    _schema = {
        'description': 'Sequential Iteration API configuration',
        'properties': {
            'infinite': {
                'description': 'Infinite iteration',
                'type': ['boolean', 'null'],
            },
            'jump': {
                'description': 'Jump entry',
                'oneOf': [{'$ref': '#/definitions/jump'}, {'type': 'null'}],
            },
            'limit': {
                'description': 'Maximum frames per task. If not passed, frames will end when no more matching frames are found, unless infinite is True.',
                'type': ['integer', 'null'],
            },
            'min_sequence': {
                'description': 'Length (in ms) of video clips to return. This is used in random order, and in sequential order only if jumping is provided and only for video frames',
                'type': ['integer', 'null'],
            },
            'order': {
                'description': "\n                Input frames order. Values: 'sequential', 'random'\n                In Sequential mode frames will be returned according to the order in which the frames were added to the dataset.",
                'type': ['string', 'null'],
            },
            'random_seed': {
                'description': 'Random seed used during iteration',
                'type': 'integer',
            },
        },
        'required': ['random_seed'],
        'type': 'object',
    }
    def __init__(
            self, random_seed, order=None, jump=None, min_sequence=None, infinite=None, limit=None, **kwargs):
        super(Iteration, self).__init__(**kwargs)
        self.order = order
        self.jump = jump
        self.min_sequence = min_sequence
        self.infinite = infinite
        self.limit = limit
        self.random_seed = random_seed

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

    @schema_property('jump')
    def jump(self):
        return self._property_jump

    @jump.setter
    def jump(self, value):
        if value is None:
            self._property_jump = None
            return
        if isinstance(value, dict):
            value = Jump.from_dict(value)
        else:
            self.assert_isinstance(value, "jump", Jump)
        self._property_jump = value

    @schema_property('min_sequence')
    def min_sequence(self):
        return self._property_min_sequence

    @min_sequence.setter
    def min_sequence(self, value):
        if value is None:
            self._property_min_sequence = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "min_sequence", six.integer_types)
        self._property_min_sequence = value

    @schema_property('infinite')
    def infinite(self):
        return self._property_infinite

    @infinite.setter
    def infinite(self, value):
        if value is None:
            self._property_infinite = None
            return
        
        self.assert_isinstance(value, "infinite", (bool,))
        self._property_infinite = value

    @schema_property('limit')
    def limit(self):
        return self._property_limit

    @limit.setter
    def limit(self, value):
        if value is None:
            self._property_limit = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "limit", six.integer_types)
        self._property_limit = value

    @schema_property('random_seed')
    def random_seed(self):
        return self._property_random_seed

    @random_seed.setter
    def random_seed(self, value):
        if value is None:
            self._property_random_seed = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "random_seed", six.integer_types)
        self._property_random_seed = value


class ViewEntry(NonStrictDataModel):
    """
    :param version: Version id of a version belonging to the dataset
    :type version: str
    :param dataset: Existing Dataset id
    :type dataset: str
    :param merge_with: Version ID to merge with
    :type merge_with: str
    """
    _schema = {
        'properties': {
            'dataset': {
                'description': 'Existing Dataset id',
                'type': ['string', 'null'],
            },
            'merge_with': {
                'description': 'Version ID to merge with',
                'type': ['string', 'null'],
            },
            'version': {
                'description': 'Version id of a version belonging to the dataset',
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, version=None, dataset=None, merge_with=None, **kwargs):
        super(ViewEntry, self).__init__(**kwargs)
        self.version = version
        self.dataset = dataset
        self.merge_with = merge_with

    @schema_property('version')
    def version(self):
        return self._property_version

    @version.setter
    def version(self, value):
        if value is None:
            self._property_version = None
            return
        
        self.assert_isinstance(value, "version", six.string_types)
        self._property_version = value

    @schema_property('dataset')
    def dataset(self):
        return self._property_dataset

    @dataset.setter
    def dataset(self, value):
        if value is None:
            self._property_dataset = None
            return
        
        self.assert_isinstance(value, "dataset", six.string_types)
        self._property_dataset = value

    @schema_property('merge_with')
    def merge_with(self):
        return self._property_merge_with

    @merge_with.setter
    def merge_with(self, value):
        if value is None:
            self._property_merge_with = None
            return
        
        self.assert_isinstance(value, "merge_with", six.string_types)
        self._property_merge_with = value


class View(NonStrictDataModel):
    """
    :param entries: List of view entries. All tasks must have at least one view.
    :type entries: Sequence[ViewEntry]
    """
    _schema = {
        'properties': {
            'entries': {
                'description': 'List of view entries. All tasks must have at least one view.',
                'items': {'$ref': '#/definitions/view_entry'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, entries=None, **kwargs):
        super(View, self).__init__(**kwargs)
        self.entries = entries

    @schema_property('entries')
    def entries(self):
        return self._property_entries

    @entries.setter
    def entries(self, value):
        if value is None:
            self._property_entries = None
            return
        
        self.assert_isinstance(value, "entries", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [ViewEntry.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "entries", ViewEntry, is_array=True)
        self._property_entries = value


class Input(NonStrictDataModel):
    """
    :param view: View params
    :type view: View
    :param frames_filter: Filtering params
    :type frames_filter: Filtering
    :param mapping: Mapping params (see common definitions section)
    :type mapping: Mapping
    :param augmentation: Augmentation parameters. Only for training and testing
        tasks.
    :type augmentation: Augmentation
    :param iteration: Iteration parameters. Not applicable for register (import)
        tasks.
    :type iteration: Iteration
    :param dataviews: Key to DataView ID Mapping
    :type dataviews: dict
    """
    _schema = {
        'properties': {
            'augmentation': {
                'description': 'Augmentation parameters. Only for training and testing tasks.',
                'oneOf': [{'$ref': '#/definitions/augmentation'}, {'type': 'null'}],
            },
            'dataviews': {
                'additionalProperties': {'type': 'string'},
                'description': 'Key to DataView ID Mapping',
                'type': ['object', 'null'],
            },
            'frames_filter': {
                'description': 'Filtering params',
                'oneOf': [{'$ref': '#/definitions/filtering'}, {'type': 'null'}],
            },
            'iteration': {
                'description': 'Iteration parameters. Not applicable for register (import) tasks.',
                'oneOf': [{'$ref': '#/definitions/iteration'}, {'type': 'null'}],
            },
            'mapping': {
                'description': 'Mapping params (see common definitions section)',
                'oneOf': [{'$ref': '#/definitions/mapping'}, {'type': 'null'}],
            },
            'view': {
                'description': 'View params',
                'oneOf': [{'$ref': '#/definitions/view'}, {'type': 'null'}],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, view=None, frames_filter=None, mapping=None, augmentation=None, iteration=None, dataviews=None, **kwargs):
        super(Input, self).__init__(**kwargs)
        self.view = view
        self.frames_filter = frames_filter
        self.mapping = mapping
        self.augmentation = augmentation
        self.iteration = iteration
        self.dataviews = dataviews

    @schema_property('view')
    def view(self):
        return self._property_view

    @view.setter
    def view(self, value):
        if value is None:
            self._property_view = None
            return
        if isinstance(value, dict):
            value = View.from_dict(value)
        else:
            self.assert_isinstance(value, "view", View)
        self._property_view = value

    @schema_property('frames_filter')
    def frames_filter(self):
        return self._property_frames_filter

    @frames_filter.setter
    def frames_filter(self, value):
        if value is None:
            self._property_frames_filter = None
            return
        if isinstance(value, dict):
            value = Filtering.from_dict(value)
        else:
            self.assert_isinstance(value, "frames_filter", Filtering)
        self._property_frames_filter = value

    @schema_property('mapping')
    def mapping(self):
        return self._property_mapping

    @mapping.setter
    def mapping(self, value):
        if value is None:
            self._property_mapping = None
            return
        if isinstance(value, dict):
            value = Mapping.from_dict(value)
        else:
            self.assert_isinstance(value, "mapping", Mapping)
        self._property_mapping = value

    @schema_property('augmentation')
    def augmentation(self):
        return self._property_augmentation

    @augmentation.setter
    def augmentation(self, value):
        if value is None:
            self._property_augmentation = None
            return
        if isinstance(value, dict):
            value = Augmentation.from_dict(value)
        else:
            self.assert_isinstance(value, "augmentation", Augmentation)
        self._property_augmentation = value

    @schema_property('iteration')
    def iteration(self):
        return self._property_iteration

    @iteration.setter
    def iteration(self, value):
        if value is None:
            self._property_iteration = None
            return
        if isinstance(value, dict):
            value = Iteration.from_dict(value)
        else:
            self.assert_isinstance(value, "iteration", Iteration)
        self._property_iteration = value

    @schema_property('dataviews')
    def dataviews(self):
        return self._property_dataviews

    @dataviews.setter
    def dataviews(self, value):
        if value is None:
            self._property_dataviews = None
            return
        
        self.assert_isinstance(value, "dataviews", (dict,))
        self._property_dataviews = value


class Output(NonStrictDataModel):
    """
    :param view: View params
    :type view: View
    :param destination: Storage id. This is where output files will be stored.
    :type destination: str
    :param model: Model id.
    :type model: str
    :param result: Task result. Values: 'success', 'failure'
    :type result: str
    :param error: Last error text
    :type error: str
    """
    _schema = {
        'properties': {
            'destination': {
                'description': 'Storage id. This is where output files will be stored.',
                'type': ['string', 'null'],
            },
            'error': {'description': 'Last error text', 'type': ['string', 'null']},
            'model': {'description': 'Model id.', 'type': ['string', 'null']},
            'result': {
                'description': "Task result. Values: 'success', 'failure'",
                'type': ['string', 'null'],
            },
            'view': {
                'description': 'View params',
                'oneOf': [{'$ref': '#/definitions/view'}, {'type': 'null'}],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, view=None, destination=None, model=None, result=None, error=None, **kwargs):
        super(Output, self).__init__(**kwargs)
        self.view = view
        self.destination = destination
        self.model = model
        self.result = result
        self.error = error

    @schema_property('view')
    def view(self):
        return self._property_view

    @view.setter
    def view(self, value):
        if value is None:
            self._property_view = None
            return
        if isinstance(value, dict):
            value = View.from_dict(value)
        else:
            self.assert_isinstance(value, "view", View)
        self._property_view = value

    @schema_property('destination')
    def destination(self):
        return self._property_destination

    @destination.setter
    def destination(self, value):
        if value is None:
            self._property_destination = None
            return
        
        self.assert_isinstance(value, "destination", six.string_types)
        self._property_destination = value

    @schema_property('model')
    def model(self):
        return self._property_model

    @model.setter
    def model(self, value):
        if value is None:
            self._property_model = None
            return
        
        self.assert_isinstance(value, "model", six.string_types)
        self._property_model = value

    @schema_property('result')
    def result(self):
        return self._property_result

    @result.setter
    def result(self, value):
        if value is None:
            self._property_result = None
            return
        
        self.assert_isinstance(value, "result", six.string_types)
        self._property_result = value

    @schema_property('error')
    def error(self):
        return self._property_error

    @error.setter
    def error(self, value):
        if value is None:
            self._property_error = None
            return
        
        self.assert_isinstance(value, "error", six.string_types)
        self._property_error = value


class OutputRoisEnum(StringEnum):
    all_in_frame = "all_in_frame"
    only_filtered = "only_filtered"
    frame_per_roi = "frame_per_roi"


class Execution(NonStrictDataModel):
    """
    :param queue: Queue ID where task was queued.
    :type queue: str
    :param test_split: Percentage of frames to use for testing only
    :type test_split: int
    :param parameters: Json object containing the Task parameters
    :type parameters: dict
    :param model: Execution input model ID Not applicable for Register (Import)
        tasks
    :type model: str
    :param model_desc: Json object representing the Model descriptors
    :type model_desc: dict
    :param model_labels: Json object representing the ids of the labels in the
        model. The keys are the layers' names and the values are the IDs. Not
        applicable for Register (Import) tasks. Mandatory for Training tasks[z]
    :type model_labels: dict
    :param framework: Framework related to the task. Case insensitive. Mandatory
        for Training tasks.
    :type framework: str
    :param dataviews: Additional dataviews for the task
    :type dataviews: Sequence[dict]
    """
    _schema = {
        'properties': {
            'dataviews': {
                'description': 'Additional dataviews for the task',
                'items': {'additionalProperties': True, 'type': 'object'},
                'type': ['array', 'null'],
            },
            'framework': {
                'description': 'Framework related to the task. Case insensitive. Mandatory for Training tasks. ',
                'type': ['string', 'null'],
            },
            'model': {
                'description': 'Execution input model ID Not applicable for Register (Import) tasks',
                'type': ['string', 'null'],
            },
            'model_desc': {
                'additionalProperties': True,
                'description': 'Json object representing the Model descriptors',
                'type': ['object', 'null'],
            },
            'model_labels': {
                'additionalProperties': {'type': 'integer'},
                'description': "Json object representing the ids of the labels in the model.\n                The keys are the layers' names and the values are the IDs.\n                Not applicable for Register (Import) tasks.\n                Mandatory for Training tasks[z]",
                'type': ['object', 'null'],
            },
            'parameters': {
                'additionalProperties': True,
                'description': 'Json object containing the Task parameters',
                'type': ['object', 'null'],
            },
            'queue': {
                'description': 'Queue ID where task was queued.',
                'type': ['string', 'null'],
            },
            'test_split': {
                'description': 'Percentage of frames to use for testing only',
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, queue=None, test_split=None, parameters=None, model=None, model_desc=None, model_labels=None, framework=None, dataviews=None, **kwargs):
        super(Execution, self).__init__(**kwargs)
        self.queue = queue
        self.test_split = test_split
        self.parameters = parameters
        self.model = model
        self.model_desc = model_desc
        self.model_labels = model_labels
        self.framework = framework
        self.dataviews = dataviews

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

    @schema_property('test_split')
    def test_split(self):
        return self._property_test_split

    @test_split.setter
    def test_split(self, value):
        if value is None:
            self._property_test_split = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "test_split", six.integer_types)
        self._property_test_split = value

    @schema_property('parameters')
    def parameters(self):
        return self._property_parameters

    @parameters.setter
    def parameters(self, value):
        if value is None:
            self._property_parameters = None
            return
        
        self.assert_isinstance(value, "parameters", (dict,))
        self._property_parameters = value

    @schema_property('model')
    def model(self):
        return self._property_model

    @model.setter
    def model(self, value):
        if value is None:
            self._property_model = None
            return
        
        self.assert_isinstance(value, "model", six.string_types)
        self._property_model = value

    @schema_property('model_desc')
    def model_desc(self):
        return self._property_model_desc

    @model_desc.setter
    def model_desc(self, value):
        if value is None:
            self._property_model_desc = None
            return
        
        self.assert_isinstance(value, "model_desc", (dict,))
        self._property_model_desc = value

    @schema_property('model_labels')
    def model_labels(self):
        return self._property_model_labels

    @model_labels.setter
    def model_labels(self, value):
        if value is None:
            self._property_model_labels = None
            return
        
        self.assert_isinstance(value, "model_labels", (dict,))
        self._property_model_labels = value

    @schema_property('framework')
    def framework(self):
        return self._property_framework

    @framework.setter
    def framework(self, value):
        if value is None:
            self._property_framework = None
            return
        
        self.assert_isinstance(value, "framework", six.string_types)
        self._property_framework = value

    @schema_property('dataviews')
    def dataviews(self):
        return self._property_dataviews

    @dataviews.setter
    def dataviews(self, value):
        if value is None:
            self._property_dataviews = None
            return
        
        self.assert_isinstance(value, "dataviews", (list, tuple))
        
        self.assert_isinstance(value, "dataviews", (dict,), is_array=True)
        self._property_dataviews = value


class TaskStatusEnum(StringEnum):
    created = "created"
    queued = "queued"
    in_progress = "in_progress"
    stopped = "stopped"
    published = "published"
    publishing = "publishing"
    closed = "closed"
    failed = "failed"
    unknown = "unknown"


class TaskTypeEnum(StringEnum):
    training = "training"
    testing = "testing"


class LastMetricsEvent(NonStrictDataModel):
    """
    :param metric: Metric name
    :type metric: str
    :param variant: Variant name
    :type variant: str
    :param type: Event type
    :type type: str
    :param timestamp: Event report time (UTC)
    :type timestamp: datetime.datetime
    :param iter: Iteration number
    :type iter: int
    :param value: Value
    :type value: float
    """
    _schema = {
        'properties': {
            'iter': {
                'description': 'Iteration number',
                'type': ['integer', 'null'],
            },
            'metric': {'description': 'Metric name', 'type': ['string', 'null']},
            'timestamp': {
                'description': 'Event report time (UTC)',
                'format': 'date-time',
                'type': ['string', 'null'],
            },
            'type': {'description': 'Event type', 'type': ['string', 'null']},
            'value': {'description': 'Value', 'type': ['number', 'null']},
            'variant': {'description': 'Variant name', 'type': ['string', 'null']},
        },
        'type': 'object',
    }
    def __init__(
            self, metric=None, variant=None, type=None, timestamp=None, iter=None, value=None, **kwargs):
        super(LastMetricsEvent, self).__init__(**kwargs)
        self.metric = metric
        self.variant = variant
        self.type = type
        self.timestamp = timestamp
        self.iter = iter
        self.value = value

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

    @schema_property('type')
    def type(self):
        return self._property_type

    @type.setter
    def type(self, value):
        if value is None:
            self._property_type = None
            return
        
        self.assert_isinstance(value, "type", six.string_types)
        self._property_type = value

    @schema_property('timestamp')
    def timestamp(self):
        return self._property_timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            self._property_timestamp = None
            return
        
        self.assert_isinstance(value, "timestamp", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_timestamp = value

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


class LastMetricsVariants(NonStrictDataModel):
    """
    Last metric events, one for each variant hash

    """
    _schema = {
        'additionalProperties': {'$ref': '#/definitions/last_metrics_event'},
        'description': 'Last metric events, one for each variant hash',
        'type': 'object',
    }


class Task(NonStrictDataModel):
    """
    :param id: Task id
    :type id: str
    :param name: Task Name
    :type name: str
    :param user: Associated user id
    :type user: str
    :param company: Company ID
    :type company: str
    :param type: Type of task. Values: 'dataset_import', 'annotation', 'training',
        'testing'
    :type type: TaskTypeEnum
    :param status:
    :type status: TaskStatusEnum
    :param comment: Free text comment
    :type comment: str
    :param created: Task creation time (UTC)
    :type created: datetime.datetime
    :param started: Task start time (UTC)
    :type started: datetime.datetime
    :param completed: Task end time (UTC)
    :type completed: datetime.datetime
    :param parent: Parent task id
    :type parent: str
    :param project: Project ID of the project to which this task is assigned
    :type project: str
    :param input: Task input params
    :type input: Input
    :param output: Task output params
    :type output: Output
    :param execution: Task execution params
    :type execution: Execution
    :param script: Script info
    :type script: Script
    :param tags: Tags list
    :type tags: Sequence[str]
    :param status_changed: Last status change time
    :type status_changed: datetime.datetime
    :param status_message: free text string representing info about the status
    :type status_message: str
    :param status_reason: Reason for last status change
    :type status_reason: str
    :param published: Last status change time
    :type published: datetime.datetime
    :param last_worker: ID of last worker that handled the task
    :type last_worker: str
    :param last_worker_report: Last time a worker reported while working on this
        task
    :type last_worker_report: datetime.datetime
    :param last_update: Last time this task was created, updated, changed or events
        for this task were reported
    :type last_update: datetime.datetime
    :param last_iteration: Last iteration reported for this task
    :type last_iteration: int
    :param last_metrics: Last metric variants (hash to events), one for each metric
        hash
    :type last_metrics: dict
    """
    _schema = {
        'properties': {
            'comment': {
                'description': 'Free text comment',
                'type': ['string', 'null'],
            },
            'company': {'description': 'Company ID', 'type': ['string', 'null']},
            'completed': {
                'description': 'Task end time (UTC)',
                'format': 'date-time',
                'type': ['string', 'null'],
            },
            'created': {
                'description': 'Task creation time (UTC) ',
                'format': 'date-time',
                'type': ['string', 'null'],
            },
            'execution': {
                'description': 'Task execution params',
                'oneOf': [{'$ref': '#/definitions/execution'}, {'type': 'null'}],
            },
            'id': {'description': 'Task id', 'type': ['string', 'null']},
            'input': {
                'description': 'Task input params',
                'oneOf': [{'$ref': '#/definitions/input'}, {'type': 'null'}],
            },
            'last_iteration': {
                'description': 'Last iteration reported for this task',
                'type': ['integer', 'null'],
            },
            'last_metrics': {
                'additionalProperties': {
                    '$ref': '#/definitions/last_metrics_variants',
                },
                'description': 'Last metric variants (hash to events), one for each metric hash',
                'type': ['object', 'null'],
            },
            'last_update': {
                'description': 'Last time this task was created, updated, changed or events for this task were reported',
                'format': 'date-time',
                'type': ['string', 'null'],
            },
            'last_worker': {
                'description': 'ID of last worker that handled the task',
                'type': ['string', 'null'],
            },
            'last_worker_report': {
                'description': 'Last time a worker reported while working on this task',
                'format': 'date-time',
                'type': ['string', 'null'],
            },
            'name': {'description': 'Task Name', 'type': ['string', 'null']},
            'output': {
                'description': 'Task output params',
                'oneOf': [{'$ref': '#/definitions/output'}, {'type': 'null'}],
            },
            'parent': {'description': 'Parent task id', 'type': ['string', 'null']},
            'project': {
                'description': 'Project ID of the project to which this task is assigned',
                'type': ['string', 'null'],
            },
            'published': {
                'description': 'Last status change time',
                'format': 'date-time',
                'type': ['string', 'null'],
            },
            'script': {
                'description': 'Script info',
                'oneOf': [{'$ref': '#/definitions/script'}, {'type': 'null'}],
            },
            'started': {
                'description': 'Task start time (UTC)',
                'format': 'date-time',
                'type': ['string', 'null'],
            },
            'status': {
                'description': '',
                'oneOf': [
                    {'$ref': '#/definitions/task_status_enum'},
                    {'type': 'null'},
                ],
            },
            'status_changed': {
                'description': 'Last status change time',
                'format': 'date-time',
                'type': ['string', 'null'],
            },
            'status_message': {
                'description': 'free text string representing info about the status',
                'type': ['string', 'null'],
            },
            'status_reason': {
                'description': 'Reason for last status change',
                'type': ['string', 'null'],
            },
            'tags': {
                'description': 'Tags list',
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
            'type': {
                'description': "Type of task. Values: 'dataset_import', 'annotation', 'training', 'testing'",
                'oneOf': [
                    {'$ref': '#/definitions/task_type_enum'},
                    {'type': 'null'},
                ],
            },
            'user': {
                'description': 'Associated user id',
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, id=None, name=None, user=None, company=None, type=None, status=None, comment=None, created=None, started=None, completed=None, parent=None, project=None, input=None, output=None, execution=None, script=None, tags=None, status_changed=None, status_message=None, status_reason=None, published=None, last_worker=None, last_worker_report=None, last_update=None, last_iteration=None, last_metrics=None, **kwargs):
        super(Task, self).__init__(**kwargs)
        self.id = id
        self.name = name
        self.user = user
        self.company = company
        self.type = type
        self.status = status
        self.comment = comment
        self.created = created
        self.started = started
        self.completed = completed
        self.parent = parent
        self.project = project
        self.input = input
        self.output = output
        self.execution = execution
        self.script = script
        self.tags = tags
        self.status_changed = status_changed
        self.status_message = status_message
        self.status_reason = status_reason
        self.published = published
        self.last_worker = last_worker
        self.last_worker_report = last_worker_report
        self.last_update = last_update
        self.last_iteration = last_iteration
        self.last_metrics = last_metrics

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

    @schema_property('user')
    def user(self):
        return self._property_user

    @user.setter
    def user(self, value):
        if value is None:
            self._property_user = None
            return
        
        self.assert_isinstance(value, "user", six.string_types)
        self._property_user = value

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

    @schema_property('type')
    def type(self):
        return self._property_type

    @type.setter
    def type(self, value):
        if value is None:
            self._property_type = None
            return
        if isinstance(value, six.string_types):
            try:
                value = TaskTypeEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "type", enum.Enum)
        self._property_type = value

    @schema_property('status')
    def status(self):
        return self._property_status

    @status.setter
    def status(self, value):
        if value is None:
            self._property_status = None
            return
        if isinstance(value, six.string_types):
            try:
                value = TaskStatusEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "status", enum.Enum)
        self._property_status = value

    @schema_property('comment')
    def comment(self):
        return self._property_comment

    @comment.setter
    def comment(self, value):
        if value is None:
            self._property_comment = None
            return
        
        self.assert_isinstance(value, "comment", six.string_types)
        self._property_comment = value

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

    @schema_property('started')
    def started(self):
        return self._property_started

    @started.setter
    def started(self, value):
        if value is None:
            self._property_started = None
            return
        
        self.assert_isinstance(value, "started", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_started = value

    @schema_property('completed')
    def completed(self):
        return self._property_completed

    @completed.setter
    def completed(self, value):
        if value is None:
            self._property_completed = None
            return
        
        self.assert_isinstance(value, "completed", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_completed = value

    @schema_property('parent')
    def parent(self):
        return self._property_parent

    @parent.setter
    def parent(self, value):
        if value is None:
            self._property_parent = None
            return
        
        self.assert_isinstance(value, "parent", six.string_types)
        self._property_parent = value

    @schema_property('project')
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return
        
        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property('input')
    def input(self):
        return self._property_input

    @input.setter
    def input(self, value):
        if value is None:
            self._property_input = None
            return
        if isinstance(value, dict):
            value = Input.from_dict(value)
        else:
            self.assert_isinstance(value, "input", Input)
        self._property_input = value

    @schema_property('output')
    def output(self):
        return self._property_output

    @output.setter
    def output(self, value):
        if value is None:
            self._property_output = None
            return
        if isinstance(value, dict):
            value = Output.from_dict(value)
        else:
            self.assert_isinstance(value, "output", Output)
        self._property_output = value

    @schema_property('execution')
    def execution(self):
        return self._property_execution

    @execution.setter
    def execution(self, value):
        if value is None:
            self._property_execution = None
            return
        if isinstance(value, dict):
            value = Execution.from_dict(value)
        else:
            self.assert_isinstance(value, "execution", Execution)
        self._property_execution = value

    @schema_property('script')
    def script(self):
        return self._property_script

    @script.setter
    def script(self, value):
        if value is None:
            self._property_script = None
            return
        if isinstance(value, dict):
            value = Script.from_dict(value)
        else:
            self.assert_isinstance(value, "script", Script)
        self._property_script = value

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

    @schema_property('status_changed')
    def status_changed(self):
        return self._property_status_changed

    @status_changed.setter
    def status_changed(self, value):
        if value is None:
            self._property_status_changed = None
            return
        
        self.assert_isinstance(value, "status_changed", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_status_changed = value

    @schema_property('status_message')
    def status_message(self):
        return self._property_status_message

    @status_message.setter
    def status_message(self, value):
        if value is None:
            self._property_status_message = None
            return
        
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value

    @schema_property('status_reason')
    def status_reason(self):
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value):
        if value is None:
            self._property_status_reason = None
            return
        
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property('published')
    def published(self):
        return self._property_published

    @published.setter
    def published(self, value):
        if value is None:
            self._property_published = None
            return
        
        self.assert_isinstance(value, "published", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_published = value

    @schema_property('last_worker')
    def last_worker(self):
        return self._property_last_worker

    @last_worker.setter
    def last_worker(self, value):
        if value is None:
            self._property_last_worker = None
            return
        
        self.assert_isinstance(value, "last_worker", six.string_types)
        self._property_last_worker = value

    @schema_property('last_worker_report')
    def last_worker_report(self):
        return self._property_last_worker_report

    @last_worker_report.setter
    def last_worker_report(self, value):
        if value is None:
            self._property_last_worker_report = None
            return
        
        self.assert_isinstance(value, "last_worker_report", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_last_worker_report = value

    @schema_property('last_update')
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

    @schema_property('last_metrics')
    def last_metrics(self):
        return self._property_last_metrics

    @last_metrics.setter
    def last_metrics(self, value):
        if value is None:
            self._property_last_metrics = None
            return
        
        self.assert_isinstance(value, "last_metrics", (dict,))
        self._property_last_metrics = value


class CloseRequest(Request):
    """
    Indicates that task is closed

    :param force: Allows forcing state change even if transition is not supported
    :type force: bool
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "close"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'force': {
                'default': False,
                'description': 'Allows forcing state change even if transition is not supported',
                'type': ['boolean', 'null'],
            },
            'status_message': {
                'description': 'Extra information regarding status change',
                'type': 'string',
            },
            'status_reason': {
                'description': 'Reason for status change',
                'type': 'string',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }
    def __init__(
            self, task, force=False, status_reason=None, status_message=None, **kwargs):
        super(CloseRequest, self).__init__(**kwargs)
        self.force = force
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property('force')
    def force(self):
        return self._property_force

    @force.setter
    def force(self, value):
        if value is None:
            self._property_force = None
            return
        
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

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

    @schema_property('status_reason')
    def status_reason(self):
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value):
        if value is None:
            self._property_status_reason = None
            return
        
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property('status_message')
    def status_message(self):
        return self._property_status_message

    @status_message.setter
    def status_message(self, value):
        if value is None:
            self._property_status_message = None
            return
        
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class CloseResponse(Response):
    """
    Response of tasks.close endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """
    _service = "tasks"
    _action = "close"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'fields': {
                'additionalProperties': True,
                'description': 'Updated fields names and values',
                'type': ['object', 'null'],
            },
            'updated': {
                'description': 'Number of tasks updated (0 or 1)',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, updated=None, fields=None, **kwargs):
        super(CloseResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property('updated')
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

    @schema_property('fields')
    def fields(self):
        return self._property_fields

    @fields.setter
    def fields(self, value):
        if value is None:
            self._property_fields = None
            return
        
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class CreateRequest(Request):
    """
    Create a new task

    :param name: Task name. Unique within the company.
    :type name: str
    :param tags: Tags list
    :type tags: Sequence[str]
    :param type: Type of task
    :type type: TaskTypeEnum
    :param comment: Free text comment
    :type comment: str
    :param parent: Parent task id Must be a completed task.
    :type parent: str
    :param project: Project ID of the project to which this task is assigned Must
        exist[ab]
    :type project: str
    :param input: Task input params.  (input view must be provided).
    :type input: Input
    :param output_dest: Output storage id Must be a reference to an existing
        storage.
    :type output_dest: str
    :param execution: Task execution params
    :type execution: Execution
    :param script: Script info
    :type script: Script
    """

    _service = "tasks"
    _action = "create"
    _version = "1.9"
    _schema = {
        'definitions': {
            'augmentation': {
                'properties': {
                    'crop_around_rois': {
                        'description': 'Crop image data around all frame ROIs',
                        'type': ['boolean', 'null'],
                    },
                    'sets': {
                        'description': 'List of augmentation sets',
                        'items': {'$ref': '#/definitions/augmentation_set'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'augmentation_set': {
                'properties': {
                    'arguments': {
                        'additionalProperties': {
                            'additionalProperties': True,
                            'type': 'object',
                        },
                        'description': 'Arguments dictionary per custom augmentation type.',
                        'type': ['object', 'null'],
                    },
                    'cls': {
                        'description': 'Augmentation class',
                        'type': ['string', 'null'],
                    },
                    'strength': {
                        'description': 'Augmentation strength. Range [0,).',
                        'minimum': 0,
                        'type': ['number', 'null'],
                    },
                    'types': {
                        'description': 'Augmentation type',
                        'items': {'type': 'string'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'execution': {
                'properties': {
                    'dataviews': {
                        'description': 'Additional dataviews for the task',
                        'items': {'additionalProperties': True, 'type': 'object'},
                        'type': ['array', 'null'],
                    },
                    'framework': {
                        'description': 'Framework related to the task. Case insensitive. Mandatory for Training tasks. ',
                        'type': ['string', 'null'],
                    },
                    'model': {
                        'description': 'Execution input model ID Not applicable for Register (Import) tasks',
                        'type': ['string', 'null'],
                    },
                    'model_desc': {
                        'additionalProperties': True,
                        'description': 'Json object representing the Model descriptors',
                        'type': ['object', 'null'],
                    },
                    'model_labels': {
                        'additionalProperties': {'type': 'integer'},
                        'description': "Json object representing the ids of the labels in the model.\n                The keys are the layers' names and the values are the IDs.\n                Not applicable for Register (Import) tasks.\n                Mandatory for Training tasks[z]",
                        'type': ['object', 'null'],
                    },
                    'parameters': {
                        'additionalProperties': True,
                        'description': 'Json object containing the Task parameters',
                        'type': ['object', 'null'],
                    },
                    'queue': {
                        'description': 'Queue ID where task was queued.',
                        'type': ['string', 'null'],
                    },
                    'test_split': {
                        'description': 'Percentage of frames to use for testing only',
                        'type': ['integer', 'null'],
                    },
                },
                'type': 'object',
            },
            'filter_by_roi_enum': {
                'default': 'label_rules',
                'enum': ['disabled', 'no_rois', 'label_rules'],
                'type': 'string',
            },
            'filter_label_rule': {
                'properties': {
                    'conf_range': {
                        'description': 'Range of ROI confidence level in the frame (min, max). -1 for not applicable\n            Both min and max can be either -1 or positive.\n            2nd number (max) must be either -1 or larger than or equal to the 1st number (min)',
                        'items': {'type': 'number'},
                        'maxItems': 2,
                        'minItems': 1,
                        'type': 'array',
                    },
                    'count_range': {
                        'description': 'Range of times ROI appears in the frame (min, max). -1 for not applicable.\n            Both integers must be larger than or equal to -1.\n            2nd integer (max) must be either -1 or larger than or equal to the 1st integer (min)',
                        'items': {'type': 'integer'},
                        'maxItems': 2,
                        'minItems': 1,
                        'type': 'array',
                    },
                    'label': {
                        'description': "Lucene format query (see lucene query syntax).\nDefault search field is label.keyword and default operator is AND, so searching for:\n\n'Bus Stop' Blue\n\nis equivalent to:\n\nLabel.keyword:'Bus Stop' AND label.keyword:'Blue'",
                        'type': 'string',
                    },
                },
                'required': ['label'],
                'type': 'object',
            },
            'filter_rule': {
                'properties': {
                    'dataset': {
                        'description': "Dataset ID. Must be a dataset which is in the task's view. If set to '*' all datasets in View are used.",
                        'type': 'string',
                    },
                    'filter_by_roi': {
                        '$ref': '#/definitions/filter_by_roi_enum',
                        'description': 'Type of filter',
                    },
                    'frame_query': {
                        'description': 'Frame filter, in Lucene query syntax',
                        'type': 'string',
                    },
                    'label_rules': {
                        'description': "List of FilterLabelRule ('AND' connection)\n\ndisabled - No filtering by ROIs. Select all frames, even if they don't have ROIs (all frames)\n\nno_rois - Select only frames without ROIs (empty frames)\n\nlabel_rules - Select frames according to label rules",
                        'items': {'$ref': '#/definitions/filter_label_rule'},
                        'type': ['array', 'null'],
                    },
                    'sources_query': {
                        'description': 'Sources filter, in Lucene query syntax. Filters sources in each frame.',
                        'type': 'string',
                    },
                    'version': {
                        'description': "Dataset version to apply rule to. Must belong to the dataset and be in the task's view. If set to '*' all version of the datasets in View are used.",
                        'type': 'string',
                    },
                    'weight': {
                        'description': 'Rule weight. Default is 1',
                        'type': 'number',
                    },
                },
                'required': ['filter_by_roi'],
                'type': 'object',
            },
            'filtering': {
                'properties': {
                    'filtering_rules': {
                        'description': "List of FilterRule ('OR' connection)",
                        'items': {'$ref': '#/definitions/filter_rule'},
                        'type': ['array', 'null'],
                    },
                    'output_rois': {
                        'description': "'all_in_frame' - all rois for a frame are returned\n\n'only_filtered' - only rois which led this frame to be selected\n\n'frame_per_roi' - single roi per frame. Frame can be returned multiple times with a different roi each time.\n\nNote: this should be used for Training tasks only\n\nNote: frame_per_roi implies that only filtered rois will be returned\n                ",
                        'oneOf': [
                            {'$ref': '#/definitions/output_rois_enum'},
                            {'type': 'null'},
                        ],
                    },
                },
                'type': 'object',
            },
            'input': {
                'properties': {
                    'augmentation': {
                        'description': 'Augmentation parameters. Only for training and testing tasks.',
                        'oneOf': [
                            {'$ref': '#/definitions/augmentation'},
                            {'type': 'null'},
                        ],
                    },
                    'dataviews': {
                        'additionalProperties': {'type': 'string'},
                        'description': 'Key to DataView ID Mapping',
                        'type': ['object', 'null'],
                    },
                    'frames_filter': {
                        'description': 'Filtering params',
                        'oneOf': [
                            {'$ref': '#/definitions/filtering'},
                            {'type': 'null'},
                        ],
                    },
                    'iteration': {
                        'description': 'Iteration parameters. Not applicable for register (import) tasks.',
                        'oneOf': [
                            {'$ref': '#/definitions/iteration'},
                            {'type': 'null'},
                        ],
                    },
                    'mapping': {
                        'description': 'Mapping params (see common definitions section)',
                        'oneOf': [
                            {'$ref': '#/definitions/mapping'},
                            {'type': 'null'},
                        ],
                    },
                    'view': {
                        'description': 'View params',
                        'oneOf': [{'$ref': '#/definitions/view'}, {'type': 'null'}],
                    },
                },
                'type': 'object',
            },
            'iteration': {
                'description': 'Sequential Iteration API configuration',
                'properties': {
                    'infinite': {
                        'description': 'Infinite iteration',
                        'type': ['boolean', 'null'],
                    },
                    'jump': {
                        'description': 'Jump entry',
                        'oneOf': [{'$ref': '#/definitions/jump'}, {'type': 'null'}],
                    },
                    'limit': {
                        'description': 'Maximum frames per task. If not passed, frames will end when no more matching frames are found, unless infinite is True.',
                        'type': ['integer', 'null'],
                    },
                    'min_sequence': {
                        'description': 'Length (in ms) of video clips to return. This is used in random order, and in sequential order only if jumping is provided and only for video frames',
                        'type': ['integer', 'null'],
                    },
                    'order': {
                        'description': "\n                Input frames order. Values: 'sequential', 'random'\n                In Sequential mode frames will be returned according to the order in which the frames were added to the dataset.",
                        'type': ['string', 'null'],
                    },
                    'random_seed': {
                        'description': 'Random seed used during iteration',
                        'type': 'integer',
                    },
                },
                'required': ['random_seed'],
                'type': 'object',
            },
            'jump': {
                'properties': {
                    'time': {
                        'description': 'Max time in milliseconds between frames',
                        'type': ['integer', 'null'],
                    },
                },
                'type': 'object',
            },
            'label_source': {
                'properties': {
                    'dataset': {
                        'description': "Source dataset id. '*' for all datasets in view",
                        'type': ['string', 'null'],
                    },
                    'labels': {
                        'description': "List of source labels (AND connection). '*' indicates any label. Labels must exist in at least one of the dataset versions in the task's view",
                        'items': {'type': 'string'},
                        'type': ['array', 'null'],
                    },
                    'version': {
                        'description': "Source dataset version id. Default is '*' (for all versions in dataset in the view) Version must belong to the selected dataset, and must be in the task's view[i]",
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'mapping': {
                'properties': {
                    'rules': {
                        'description': 'Rules list',
                        'items': {'$ref': '#/definitions/mapping_rule'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'mapping_rule': {
                'properties': {
                    'source': {
                        'description': 'Source label info',
                        'oneOf': [
                            {'$ref': '#/definitions/label_source'},
                            {'type': 'null'},
                        ],
                    },
                    'target': {
                        'description': 'Target label name',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'output_rois_enum': {
                'enum': ['all_in_frame', 'only_filtered', 'frame_per_roi'],
                'type': 'string',
            },
            'script': {
                'properties': {
                    'binary': {
                        'default': 'python',
                        'description': 'Binary to use when running the script',
                        'type': ['string', 'null'],
                    },
                    'branch': {
                        'description': 'Repository branch id If not provided and tag not provided, default repository branch is used.',
                        'type': ['string', 'null'],
                    },
                    'entry_point': {
                        'description': 'Path to execute within the repository',
                        'type': ['string', 'null'],
                    },
                    'repository': {
                        'description': 'Name of the repository where the script is located',
                        'type': ['string', 'null'],
                    },
                    'requirements': {
                        'description': 'A JSON object containing requirements strings by key',
                        'type': ['object', 'null'],
                    },
                    'tag': {
                        'description': 'Repository tag',
                        'type': ['string', 'null'],
                    },
                    'version_num': {
                        'description': 'Version (changeset) number. Optional (default is head version) Unused if tag is provided.',
                        'type': ['string', 'null'],
                    },
                    'working_dir': {
                        'description': 'Path to the folder from which to run the script Default - root folder of repository[f]',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'task_type_enum': {
                'enum': [
                    'dataset_import',
                    'annotation',
                    'annotation_manual',
                    'training',
                    'testing',
                ],
                'type': 'string',
            },
            'view': {
                'properties': {
                    'entries': {
                        'description': 'List of view entries. All tasks must have at least one view.',
                        'items': {'$ref': '#/definitions/view_entry'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'view_entry': {
                'properties': {
                    'dataset': {
                        'description': 'Existing Dataset id',
                        'type': ['string', 'null'],
                    },
                    'merge_with': {
                        'description': 'Version ID to merge with',
                        'type': ['string', 'null'],
                    },
                    'version': {
                        'description': 'Version id of a version belonging to the dataset',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
        },
        'properties': {
            'comment': {'description': 'Free text comment ', 'type': 'string'},
            'execution': {
                '$ref': '#/definitions/execution',
                'description': 'Task execution params',
            },
            'input': {
                '$ref': '#/definitions/input',
                'description': 'Task input params.  (input view must be provided).',
            },
            'name': {
                'description': 'Task name. Unique within the company.',
                'type': 'string',
            },
            'output_dest': {
                'description': 'Output storage id Must be a reference to an existing storage.',
                'type': 'string',
            },
            'parent': {
                'description': 'Parent task id Must be a completed task.',
                'type': 'string',
            },
            'project': {
                'description': 'Project ID of the project to which this task is assigned Must exist[ab]',
                'type': 'string',
            },
            'script': {
                '$ref': '#/definitions/script',
                'description': 'Script info',
            },
            'tags': {'description': 'Tags list', 'items': {'type': 'string'}, 'type': 'array'},
            'type': {
                '$ref': '#/definitions/task_type_enum',
                'description': 'Type of task',
            },
        },
        'required': ['name', 'type'],
        'type': 'object',
    }
    def __init__(
            self, name, type, tags=None, comment=None, parent=None, project=None, input=None, output_dest=None, execution=None, script=None, **kwargs):
        super(CreateRequest, self).__init__(**kwargs)
        self.name = name
        self.tags = tags
        self.type = type
        self.comment = comment
        self.parent = parent
        self.project = project
        self.input = input
        self.output_dest = output_dest
        self.execution = execution
        self.script = script

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

    @schema_property('type')
    def type(self):
        return self._property_type

    @type.setter
    def type(self, value):
        if value is None:
            self._property_type = None
            return
        if isinstance(value, six.string_types):
            try:
                value = TaskTypeEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "type", enum.Enum)
        self._property_type = value

    @schema_property('comment')
    def comment(self):
        return self._property_comment

    @comment.setter
    def comment(self, value):
        if value is None:
            self._property_comment = None
            return
        
        self.assert_isinstance(value, "comment", six.string_types)
        self._property_comment = value

    @schema_property('parent')
    def parent(self):
        return self._property_parent

    @parent.setter
    def parent(self, value):
        if value is None:
            self._property_parent = None
            return
        
        self.assert_isinstance(value, "parent", six.string_types)
        self._property_parent = value

    @schema_property('project')
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return
        
        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property('input')
    def input(self):
        return self._property_input

    @input.setter
    def input(self, value):
        if value is None:
            self._property_input = None
            return
        if isinstance(value, dict):
            value = Input.from_dict(value)
        else:
            self.assert_isinstance(value, "input", Input)
        self._property_input = value

    @schema_property('output_dest')
    def output_dest(self):
        return self._property_output_dest

    @output_dest.setter
    def output_dest(self, value):
        if value is None:
            self._property_output_dest = None
            return
        
        self.assert_isinstance(value, "output_dest", six.string_types)
        self._property_output_dest = value

    @schema_property('execution')
    def execution(self):
        return self._property_execution

    @execution.setter
    def execution(self, value):
        if value is None:
            self._property_execution = None
            return
        if isinstance(value, dict):
            value = Execution.from_dict(value)
        else:
            self.assert_isinstance(value, "execution", Execution)
        self._property_execution = value

    @schema_property('script')
    def script(self):
        return self._property_script

    @script.setter
    def script(self, value):
        if value is None:
            self._property_script = None
            return
        if isinstance(value, dict):
            value = Script.from_dict(value)
        else:
            self.assert_isinstance(value, "script", Script)
        self._property_script = value


class CreateResponse(Response):
    """
    Response of tasks.create endpoint.

    :param id: ID of the task
    :type id: str
    """
    _service = "tasks"
    _action = "create"
    _version = "1.9"

    _schema = {
        'definitions': {},
        'properties': {
            'id': {'description': 'ID of the task', 'type': ['string', 'null']},
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
    Delete a task along with any information stored for it (statistics, frame updates etc.)
            Unless Force flag is provided, operation will fail if task has objects associated with it - i.e. children tasks, projects or datasets.
            Models that refer to the deleted task will be updated with a task ID indicating a deleted task.
            

    :param move_to_trash: Move task to trash instead of deleting it. For internal
        use only, tasks in the trash are not visible from the API and cannot be
        restored!
    :type move_to_trash: bool
    :param force: If not true, call fails if the task status is 'in_progress'
    :type force: bool
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "delete"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'force': {
                'default': False,
                'description': "If not true, call fails if the task status is 'in_progress'",
                'type': ['boolean', 'null'],
            },
            'move_to_trash': {
                'default': False,
                'description': 'Move task to trash instead of deleting it. For internal use only, tasks in the trash are not visible from the API and cannot be restored!',
                'type': ['boolean', 'null'],
            },
            'status_message': {
                'description': 'Extra information regarding status change',
                'type': 'string',
            },
            'status_reason': {
                'description': 'Reason for status change',
                'type': 'string',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }
    def __init__(
            self, task, move_to_trash=False, force=False, status_reason=None, status_message=None, **kwargs):
        super(DeleteRequest, self).__init__(**kwargs)
        self.move_to_trash = move_to_trash
        self.force = force
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property('move_to_trash')
    def move_to_trash(self):
        return self._property_move_to_trash

    @move_to_trash.setter
    def move_to_trash(self, value):
        if value is None:
            self._property_move_to_trash = None
            return
        
        self.assert_isinstance(value, "move_to_trash", (bool,))
        self._property_move_to_trash = value

    @schema_property('force')
    def force(self):
        return self._property_force

    @force.setter
    def force(self, value):
        if value is None:
            self._property_force = None
            return
        
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

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

    @schema_property('status_reason')
    def status_reason(self):
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value):
        if value is None:
            self._property_status_reason = None
            return
        
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property('status_message')
    def status_message(self):
        return self._property_status_message

    @status_message.setter
    def status_message(self, value):
        if value is None:
            self._property_status_message = None
            return
        
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class DeleteResponse(Response):
    """
    Response of tasks.delete endpoint.

    :param deleted: Indicates whether the task was deleted
    :type deleted: bool
    :param updated_children: Number of child tasks whose parent property was
        updated
    :type updated_children: int
    :param updated_models: Number of models whose task property was updated
    :type updated_models: int
    :param updated_versions: Number of dataset versions whose task property was
        updated
    :type updated_versions: int
    :param frames: Response from frames.rollback
    :type frames: dict
    :param events: Response from events.delete_for_task
    :type events: dict
    """
    _service = "tasks"
    _action = "delete"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'deleted': {
                'description': 'Indicates whether the task was deleted',
                'type': ['boolean', 'null'],
            },
            'events': {
                'additionalProperties': True,
                'description': 'Response from events.delete_for_task',
                'type': ['object', 'null'],
            },
            'frames': {
                'additionalProperties': True,
                'description': 'Response from frames.rollback',
                'type': ['object', 'null'],
            },
            'updated_children': {
                'description': 'Number of child tasks whose parent property was updated',
                'type': ['integer', 'null'],
            },
            'updated_models': {
                'description': 'Number of models whose task property was updated',
                'type': ['integer', 'null'],
            },
            'updated_versions': {
                'description': 'Number of dataset versions whose task property was updated',
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, deleted=None, updated_children=None, updated_models=None, updated_versions=None, frames=None, events=None, **kwargs):
        super(DeleteResponse, self).__init__(**kwargs)
        self.deleted = deleted
        self.updated_children = updated_children
        self.updated_models = updated_models
        self.updated_versions = updated_versions
        self.frames = frames
        self.events = events

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

    @schema_property('updated_children')
    def updated_children(self):
        return self._property_updated_children

    @updated_children.setter
    def updated_children(self, value):
        if value is None:
            self._property_updated_children = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "updated_children", six.integer_types)
        self._property_updated_children = value

    @schema_property('updated_models')
    def updated_models(self):
        return self._property_updated_models

    @updated_models.setter
    def updated_models(self, value):
        if value is None:
            self._property_updated_models = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "updated_models", six.integer_types)
        self._property_updated_models = value

    @schema_property('updated_versions')
    def updated_versions(self):
        return self._property_updated_versions

    @updated_versions.setter
    def updated_versions(self, value):
        if value is None:
            self._property_updated_versions = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "updated_versions", six.integer_types)
        self._property_updated_versions = value

    @schema_property('frames')
    def frames(self):
        return self._property_frames

    @frames.setter
    def frames(self, value):
        if value is None:
            self._property_frames = None
            return
        
        self.assert_isinstance(value, "frames", (dict,))
        self._property_frames = value

    @schema_property('events')
    def events(self):
        return self._property_events

    @events.setter
    def events(self, value):
        if value is None:
            self._property_events = None
            return
        
        self.assert_isinstance(value, "events", (dict,))
        self._property_events = value


class DequeueRequest(Request):
    """
    Remove a task from its queue.
            Fails if task status is not queued.

    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "dequeue"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'status_message': {
                'description': 'Extra information regarding status change',
                'type': 'string',
            },
            'status_reason': {
                'description': 'Reason for status change',
                'type': 'string',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }
    def __init__(
            self, task, status_reason=None, status_message=None, **kwargs):
        super(DequeueRequest, self).__init__(**kwargs)
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

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

    @schema_property('status_reason')
    def status_reason(self):
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value):
        if value is None:
            self._property_status_reason = None
            return
        
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property('status_message')
    def status_message(self):
        return self._property_status_message

    @status_message.setter
    def status_message(self, value):
        if value is None:
            self._property_status_message = None
            return
        
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class DequeueResponse(Response):
    """
    Response of tasks.dequeue endpoint.

    :param dequeued: Number of tasks dequeued (0 or 1)
    :type dequeued: int
    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """
    _service = "tasks"
    _action = "dequeue"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'dequeued': {
                'description': 'Number of tasks dequeued (0 or 1)',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
            'fields': {
                'additionalProperties': True,
                'description': 'Updated fields names and values',
                'type': ['object', 'null'],
            },
            'updated': {
                'description': 'Number of tasks updated (0 or 1)',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, dequeued=None, updated=None, fields=None, **kwargs):
        super(DequeueResponse, self).__init__(**kwargs)
        self.dequeued = dequeued
        self.updated = updated
        self.fields = fields

    @schema_property('dequeued')
    def dequeued(self):
        return self._property_dequeued

    @dequeued.setter
    def dequeued(self, value):
        if value is None:
            self._property_dequeued = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "dequeued", six.integer_types)
        self._property_dequeued = value

    @schema_property('updated')
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

    @schema_property('fields')
    def fields(self):
        return self._property_fields

    @fields.setter
    def fields(self, value):
        if value is None:
            self._property_fields = None
            return
        
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class EditRequest(Request):
    """
    Edit task's details.

    :param task: ID of the task
    :type task: str
    :param force: If not true, call fails if the task status is not 'created'
    :type force: bool
    :param name: Task name Unique within the company.
    :type name: str
    :param tags: Tags list
    :type tags: Sequence[str]
    :param type: Type of task
    :type type: TaskTypeEnum
    :param comment: Free text comment
    :type comment: str
    :param parent: Parent task id Must be a completed task.
    :type parent: str
    :param project: Project ID of the project to which this task is assigned Must
        exist[ab]
    :type project: str
    :param input: Task input params.  (input view must be provided).
    :type input: Input
    :param output_dest: Output storage id Must be a reference to an existing
        storage.
    :type output_dest: str
    :param execution: Task execution params
    :type execution: Execution
    :param script: Script info
    :type script: Script
    """

    _service = "tasks"
    _action = "edit"
    _version = "1.9"
    _schema = {
        'definitions': {
            'augmentation': {
                'properties': {
                    'crop_around_rois': {
                        'description': 'Crop image data around all frame ROIs',
                        'type': ['boolean', 'null'],
                    },
                    'sets': {
                        'description': 'List of augmentation sets',
                        'items': {'$ref': '#/definitions/augmentation_set'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'augmentation_set': {
                'properties': {
                    'arguments': {
                        'additionalProperties': {
                            'additionalProperties': True,
                            'type': 'object',
                        },
                        'description': 'Arguments dictionary per custom augmentation type.',
                        'type': ['object', 'null'],
                    },
                    'cls': {
                        'description': 'Augmentation class',
                        'type': ['string', 'null'],
                    },
                    'strength': {
                        'description': 'Augmentation strength. Range [0,).',
                        'minimum': 0,
                        'type': ['number', 'null'],
                    },
                    'types': {
                        'description': 'Augmentation type',
                        'items': {'type': 'string'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'execution': {
                'properties': {
                    'dataviews': {
                        'description': 'Additional dataviews for the task',
                        'items': {'additionalProperties': True, 'type': 'object'},
                        'type': ['array', 'null'],
                    },
                    'framework': {
                        'description': 'Framework related to the task. Case insensitive. Mandatory for Training tasks. ',
                        'type': ['string', 'null'],
                    },
                    'model': {
                        'description': 'Execution input model ID Not applicable for Register (Import) tasks',
                        'type': ['string', 'null'],
                    },
                    'model_desc': {
                        'additionalProperties': True,
                        'description': 'Json object representing the Model descriptors',
                        'type': ['object', 'null'],
                    },
                    'model_labels': {
                        'additionalProperties': {'type': 'integer'},
                        'description': "Json object representing the ids of the labels in the model.\n                The keys are the layers' names and the values are the IDs.\n                Not applicable for Register (Import) tasks.\n                Mandatory for Training tasks[z]",
                        'type': ['object', 'null'],
                    },
                    'parameters': {
                        'additionalProperties': True,
                        'description': 'Json object containing the Task parameters',
                        'type': ['object', 'null'],
                    },
                    'queue': {
                        'description': 'Queue ID where task was queued.',
                        'type': ['string', 'null'],
                    },
                    'test_split': {
                        'description': 'Percentage of frames to use for testing only',
                        'type': ['integer', 'null'],
                    },
                },
                'type': 'object',
            },
            'filter_by_roi_enum': {
                'default': 'label_rules',
                'enum': ['disabled', 'no_rois', 'label_rules'],
                'type': 'string',
            },
            'filter_label_rule': {
                'properties': {
                    'conf_range': {
                        'description': 'Range of ROI confidence level in the frame (min, max). -1 for not applicable\n            Both min and max can be either -1 or positive.\n            2nd number (max) must be either -1 or larger than or equal to the 1st number (min)',
                        'items': {'type': 'number'},
                        'maxItems': 2,
                        'minItems': 1,
                        'type': 'array',
                    },
                    'count_range': {
                        'description': 'Range of times ROI appears in the frame (min, max). -1 for not applicable.\n            Both integers must be larger than or equal to -1.\n            2nd integer (max) must be either -1 or larger than or equal to the 1st integer (min)',
                        'items': {'type': 'integer'},
                        'maxItems': 2,
                        'minItems': 1,
                        'type': 'array',
                    },
                    'label': {
                        'description': "Lucene format query (see lucene query syntax).\nDefault search field is label.keyword and default operator is AND, so searching for:\n\n'Bus Stop' Blue\n\nis equivalent to:\n\nLabel.keyword:'Bus Stop' AND label.keyword:'Blue'",
                        'type': 'string',
                    },
                },
                'required': ['label'],
                'type': 'object',
            },
            'filter_rule': {
                'properties': {
                    'dataset': {
                        'description': "Dataset ID. Must be a dataset which is in the task's view. If set to '*' all datasets in View are used.",
                        'type': 'string',
                    },
                    'filter_by_roi': {
                        '$ref': '#/definitions/filter_by_roi_enum',
                        'description': 'Type of filter',
                    },
                    'frame_query': {
                        'description': 'Frame filter, in Lucene query syntax',
                        'type': 'string',
                    },
                    'label_rules': {
                        'description': "List of FilterLabelRule ('AND' connection)\n\ndisabled - No filtering by ROIs. Select all frames, even if they don't have ROIs (all frames)\n\nno_rois - Select only frames without ROIs (empty frames)\n\nlabel_rules - Select frames according to label rules",
                        'items': {'$ref': '#/definitions/filter_label_rule'},
                        'type': ['array', 'null'],
                    },
                    'sources_query': {
                        'description': 'Sources filter, in Lucene query syntax. Filters sources in each frame.',
                        'type': 'string',
                    },
                    'version': {
                        'description': "Dataset version to apply rule to. Must belong to the dataset and be in the task's view. If set to '*' all version of the datasets in View are used.",
                        'type': 'string',
                    },
                    'weight': {
                        'description': 'Rule weight. Default is 1',
                        'type': 'number',
                    },
                },
                'required': ['filter_by_roi'],
                'type': 'object',
            },
            'filtering': {
                'properties': {
                    'filtering_rules': {
                        'description': "List of FilterRule ('OR' connection)",
                        'items': {'$ref': '#/definitions/filter_rule'},
                        'type': ['array', 'null'],
                    },
                    'output_rois': {
                        'description': "'all_in_frame' - all rois for a frame are returned\n\n'only_filtered' - only rois which led this frame to be selected\n\n'frame_per_roi' - single roi per frame. Frame can be returned multiple times with a different roi each time.\n\nNote: this should be used for Training tasks only\n\nNote: frame_per_roi implies that only filtered rois will be returned\n                ",
                        'oneOf': [
                            {'$ref': '#/definitions/output_rois_enum'},
                            {'type': 'null'},
                        ],
                    },
                },
                'type': 'object',
            },
            'input': {
                'properties': {
                    'augmentation': {
                        'description': 'Augmentation parameters. Only for training and testing tasks.',
                        'oneOf': [
                            {'$ref': '#/definitions/augmentation'},
                            {'type': 'null'},
                        ],
                    },
                    'dataviews': {
                        'additionalProperties': {'type': 'string'},
                        'description': 'Key to DataView ID Mapping',
                        'type': ['object', 'null'],
                    },
                    'frames_filter': {
                        'description': 'Filtering params',
                        'oneOf': [
                            {'$ref': '#/definitions/filtering'},
                            {'type': 'null'},
                        ],
                    },
                    'iteration': {
                        'description': 'Iteration parameters. Not applicable for register (import) tasks.',
                        'oneOf': [
                            {'$ref': '#/definitions/iteration'},
                            {'type': 'null'},
                        ],
                    },
                    'mapping': {
                        'description': 'Mapping params (see common definitions section)',
                        'oneOf': [
                            {'$ref': '#/definitions/mapping'},
                            {'type': 'null'},
                        ],
                    },
                    'view': {
                        'description': 'View params',
                        'oneOf': [{'$ref': '#/definitions/view'}, {'type': 'null'}],
                    },
                },
                'type': 'object',
            },
            'iteration': {
                'description': 'Sequential Iteration API configuration',
                'properties': {
                    'infinite': {
                        'description': 'Infinite iteration',
                        'type': ['boolean', 'null'],
                    },
                    'jump': {
                        'description': 'Jump entry',
                        'oneOf': [{'$ref': '#/definitions/jump'}, {'type': 'null'}],
                    },
                    'limit': {
                        'description': 'Maximum frames per task. If not passed, frames will end when no more matching frames are found, unless infinite is True.',
                        'type': ['integer', 'null'],
                    },
                    'min_sequence': {
                        'description': 'Length (in ms) of video clips to return. This is used in random order, and in sequential order only if jumping is provided and only for video frames',
                        'type': ['integer', 'null'],
                    },
                    'order': {
                        'description': "\n                Input frames order. Values: 'sequential', 'random'\n                In Sequential mode frames will be returned according to the order in which the frames were added to the dataset.",
                        'type': ['string', 'null'],
                    },
                    'random_seed': {
                        'description': 'Random seed used during iteration',
                        'type': 'integer',
                    },
                },
                'required': ['random_seed'],
                'type': 'object',
            },
            'jump': {
                'properties': {
                    'time': {
                        'description': 'Max time in milliseconds between frames',
                        'type': ['integer', 'null'],
                    },
                },
                'type': 'object',
            },
            'label_source': {
                'properties': {
                    'dataset': {
                        'description': "Source dataset id. '*' for all datasets in view",
                        'type': ['string', 'null'],
                    },
                    'labels': {
                        'description': "List of source labels (AND connection). '*' indicates any label. Labels must exist in at least one of the dataset versions in the task's view",
                        'items': {'type': 'string'},
                        'type': ['array', 'null'],
                    },
                    'version': {
                        'description': "Source dataset version id. Default is '*' (for all versions in dataset in the view) Version must belong to the selected dataset, and must be in the task's view[i]",
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'mapping': {
                'properties': {
                    'rules': {
                        'description': 'Rules list',
                        'items': {'$ref': '#/definitions/mapping_rule'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'mapping_rule': {
                'properties': {
                    'source': {
                        'description': 'Source label info',
                        'oneOf': [
                            {'$ref': '#/definitions/label_source'},
                            {'type': 'null'},
                        ],
                    },
                    'target': {
                        'description': 'Target label name',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'output_rois_enum': {
                'enum': ['all_in_frame', 'only_filtered', 'frame_per_roi'],
                'type': 'string',
            },
            'script': {
                'properties': {
                    'binary': {
                        'default': 'python',
                        'description': 'Binary to use when running the script',
                        'type': ['string', 'null'],
                    },
                    'branch': {
                        'description': 'Repository branch id If not provided and tag not provided, default repository branch is used.',
                        'type': ['string', 'null'],
                    },
                    'entry_point': {
                        'description': 'Path to execute within the repository',
                        'type': ['string', 'null'],
                    },
                    'repository': {
                        'description': 'Name of the repository where the script is located',
                        'type': ['string', 'null'],
                    },
                    'requirements': {
                        'description': 'A JSON object containing requirements strings by key',
                        'type': ['object', 'null'],
                    },
                    'tag': {
                        'description': 'Repository tag',
                        'type': ['string', 'null'],
                    },
                    'version_num': {
                        'description': 'Version (changeset) number. Optional (default is head version) Unused if tag is provided.',
                        'type': ['string', 'null'],
                    },
                    'working_dir': {
                        'description': 'Path to the folder from which to run the script Default - root folder of repository[f]',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'task_type_enum': {
                'enum': [
                    'dataset_import',
                    'annotation',
                    'annotation_manual',
                    'training',
                    'testing',
                ],
                'type': 'string',
            },
            'view': {
                'properties': {
                    'entries': {
                        'description': 'List of view entries. All tasks must have at least one view.',
                        'items': {'$ref': '#/definitions/view_entry'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'view_entry': {
                'properties': {
                    'dataset': {
                        'description': 'Existing Dataset id',
                        'type': ['string', 'null'],
                    },
                    'merge_with': {
                        'description': 'Version ID to merge with',
                        'type': ['string', 'null'],
                    },
                    'version': {
                        'description': 'Version id of a version belonging to the dataset',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
        },
        'properties': {
            'comment': {'description': 'Free text comment ', 'type': 'string'},
            'execution': {
                '$ref': '#/definitions/execution',
                'description': 'Task execution params',
            },
            'force': {
                'default': False,
                'description': "If not true, call fails if the task status is not 'created'",
                'type': 'boolean',
            },
            'input': {
                '$ref': '#/definitions/input',
                'description': 'Task input params.  (input view must be provided).',
            },
            'name': {
                'description': 'Task name Unique within the company.',
                'type': 'string',
            },
            'output_dest': {
                'description': 'Output storage id Must be a reference to an existing storage.',
                'type': 'string',
            },
            'parent': {
                'description': 'Parent task id Must be a completed task.',
                'type': 'string',
            },
            'project': {
                'description': 'Project ID of the project to which this task is assigned Must exist[ab]',
                'type': 'string',
            },
            'script': {
                '$ref': '#/definitions/script',
                'description': 'Script info',
            },
            'tags': {'description': 'Tags list', 'items': {'type': 'string'}, 'type': 'array'},
            'task': {'description': 'ID of the task', 'type': 'string'},
            'type': {
                '$ref': '#/definitions/task_type_enum',
                'description': 'Type of task',
            },
        },
        'required': ['task'],
        'type': 'object',
    }
    def __init__(
            self, task, force=False, name=None, tags=None, type=None, comment=None, parent=None, project=None, input=None, output_dest=None, execution=None, script=None, **kwargs):
        super(EditRequest, self).__init__(**kwargs)
        self.task = task
        self.force = force
        self.name = name
        self.tags = tags
        self.type = type
        self.comment = comment
        self.parent = parent
        self.project = project
        self.input = input
        self.output_dest = output_dest
        self.execution = execution
        self.script = script

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

    @schema_property('force')
    def force(self):
        return self._property_force

    @force.setter
    def force(self, value):
        if value is None:
            self._property_force = None
            return
        
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

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

    @schema_property('type')
    def type(self):
        return self._property_type

    @type.setter
    def type(self, value):
        if value is None:
            self._property_type = None
            return
        if isinstance(value, six.string_types):
            try:
                value = TaskTypeEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "type", enum.Enum)
        self._property_type = value

    @schema_property('comment')
    def comment(self):
        return self._property_comment

    @comment.setter
    def comment(self, value):
        if value is None:
            self._property_comment = None
            return
        
        self.assert_isinstance(value, "comment", six.string_types)
        self._property_comment = value

    @schema_property('parent')
    def parent(self):
        return self._property_parent

    @parent.setter
    def parent(self, value):
        if value is None:
            self._property_parent = None
            return
        
        self.assert_isinstance(value, "parent", six.string_types)
        self._property_parent = value

    @schema_property('project')
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return
        
        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property('input')
    def input(self):
        return self._property_input

    @input.setter
    def input(self, value):
        if value is None:
            self._property_input = None
            return
        if isinstance(value, dict):
            value = Input.from_dict(value)
        else:
            self.assert_isinstance(value, "input", Input)
        self._property_input = value

    @schema_property('output_dest')
    def output_dest(self):
        return self._property_output_dest

    @output_dest.setter
    def output_dest(self, value):
        if value is None:
            self._property_output_dest = None
            return
        
        self.assert_isinstance(value, "output_dest", six.string_types)
        self._property_output_dest = value

    @schema_property('execution')
    def execution(self):
        return self._property_execution

    @execution.setter
    def execution(self, value):
        if value is None:
            self._property_execution = None
            return
        if isinstance(value, dict):
            value = Execution.from_dict(value)
        else:
            self.assert_isinstance(value, "execution", Execution)
        self._property_execution = value

    @schema_property('script')
    def script(self):
        return self._property_script

    @script.setter
    def script(self, value):
        if value is None:
            self._property_script = None
            return
        if isinstance(value, dict):
            value = Script.from_dict(value)
        else:
            self.assert_isinstance(value, "script", Script)
        self._property_script = value


class EditResponse(Response):
    """
    Response of tasks.edit endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """
    _service = "tasks"
    _action = "edit"
    _version = "1.9"

    _schema = {
        'definitions': {},
        'properties': {
            'fields': {
                'additionalProperties': True,
                'description': 'Updated fields names and values',
                'type': ['object', 'null'],
            },
            'updated': {
                'description': 'Number of tasks updated (0 or 1)',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, updated=None, fields=None, **kwargs):
        super(EditResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property('updated')
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

    @schema_property('fields')
    def fields(self):
        return self._property_fields

    @fields.setter
    def fields(self, value):
        if value is None:
            self._property_fields = None
            return
        
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class EnqueueRequest(Request):
    """
    Adds a task into a queue.

    Fails if task state is not 'created'.

    Fails if the following parameters in the task were not filled:

    * execution.script.repository

    * execution.script.entrypoint


    :param queue: Queue id. If not provided, task is added to the default queue.
    :type queue: str
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "enqueue"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'queue': {
                'description': 'Queue id. If not provided, task is added to the default queue.',
                'type': ['string', 'null'],
            },
            'status_message': {
                'description': 'Extra information regarding status change',
                'type': 'string',
            },
            'status_reason': {
                'description': 'Reason for status change',
                'type': 'string',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }
    def __init__(
            self, task, queue=None, status_reason=None, status_message=None, **kwargs):
        super(EnqueueRequest, self).__init__(**kwargs)
        self.queue = queue
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

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

    @schema_property('status_reason')
    def status_reason(self):
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value):
        if value is None:
            self._property_status_reason = None
            return
        
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property('status_message')
    def status_message(self):
        return self._property_status_message

    @status_message.setter
    def status_message(self, value):
        if value is None:
            self._property_status_message = None
            return
        
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class EnqueueResponse(Response):
    """
    Response of tasks.enqueue endpoint.

    :param queued: Number of tasks queued (0 or 1)
    :type queued: int
    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """
    _service = "tasks"
    _action = "enqueue"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'fields': {
                'additionalProperties': True,
                'description': 'Updated fields names and values',
                'type': ['object', 'null'],
            },
            'queued': {
                'description': 'Number of tasks queued (0 or 1)',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
            'updated': {
                'description': 'Number of tasks updated (0 or 1)',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, queued=None, updated=None, fields=None, **kwargs):
        super(EnqueueResponse, self).__init__(**kwargs)
        self.queued = queued
        self.updated = updated
        self.fields = fields

    @schema_property('queued')
    def queued(self):
        return self._property_queued

    @queued.setter
    def queued(self, value):
        if value is None:
            self._property_queued = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "queued", six.integer_types)
        self._property_queued = value

    @schema_property('updated')
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

    @schema_property('fields')
    def fields(self):
        return self._property_fields

    @fields.setter
    def fields(self, value):
        if value is None:
            self._property_fields = None
            return
        
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class FailedRequest(Request):
    """
    Indicates that task has failed

    :param force: Allows forcing state change even if transition is not supported
    :type force: bool
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "failed"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'force': {
                'default': False,
                'description': 'Allows forcing state change even if transition is not supported',
                'type': ['boolean', 'null'],
            },
            'status_message': {
                'description': 'Extra information regarding status change',
                'type': 'string',
            },
            'status_reason': {
                'description': 'Reason for status change',
                'type': 'string',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }
    def __init__(
            self, task, force=False, status_reason=None, status_message=None, **kwargs):
        super(FailedRequest, self).__init__(**kwargs)
        self.force = force
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property('force')
    def force(self):
        return self._property_force

    @force.setter
    def force(self, value):
        if value is None:
            self._property_force = None
            return
        
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

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

    @schema_property('status_reason')
    def status_reason(self):
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value):
        if value is None:
            self._property_status_reason = None
            return
        
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property('status_message')
    def status_message(self):
        return self._property_status_message

    @status_message.setter
    def status_message(self, value):
        if value is None:
            self._property_status_message = None
            return
        
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class FailedResponse(Response):
    """
    Response of tasks.failed endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """
    _service = "tasks"
    _action = "failed"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'fields': {
                'additionalProperties': True,
                'description': 'Updated fields names and values',
                'type': ['object', 'null'],
            },
            'updated': {
                'description': 'Number of tasks updated (0 or 1)',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, updated=None, fields=None, **kwargs):
        super(FailedResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property('updated')
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

    @schema_property('fields')
    def fields(self):
        return self._property_fields

    @fields.setter
    def fields(self, value):
        if value is None:
            self._property_fields = None
            return
        
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class GetAllRequest(Request):
    """
    Get all the company's tasks and all public tasks

    :param id: List of IDs to filter by
    :type id: Sequence[str]
    :param name: Get only tasks whose name matches this pattern (python regular
        expression syntax)
    :type name: str
    :param user: List of user IDs used to filter results by the task's creating
        user
    :type user: Sequence[str]
    :param project: List of project IDs
    :type project: Sequence[str]
    :param page: Page number, returns a specific page out of the resulting list of
        tasks
    :type page: int
    :param page_size: Page size, specifies the number of results returned in each
        page (last page may contain fewer results)
    :type page_size: int
    :param order_by: List of field names to order by. When search_text is used,
        '@text_score' can be used as a field representing the text score of returned
        documents. Use '-' prefix to specify descending order. Optional, recommended
        when using page
    :type order_by: Sequence[str]
    :param type: List of task types. One or more of: 'import', 'annotation',
        'training' or 'testing' (case insensitive)
    :type type: Sequence[str]
    :param tags: List of task tags. Use '-' prefix to exclude tags
    :type tags: Sequence[str]
    :param status: List of task status.
    :type status: Sequence[TaskStatusEnum]
    :param only_fields: List of task field names (nesting is supported using '.',
        e.g. execution.model_labels). If provided, this list defines the query's
        projection (only these fields will be returned for each result entry)
    :type only_fields: Sequence[str]
    :param parent: Parent ID
    :type parent: str
    :param status_changed: List of status changed constraint strings (utcformat,
        epoch) with an optional prefix modifier (>, >=, <, <=)
    :type status_changed: Sequence[str]
    :param search_text: Free text search query
    :type search_text: str
    :param _all_: Multi-field pattern condition (all fields match pattern)
    :type _all_: MultiFieldPatternData
    :param _any_: Multi-field pattern condition (any field matches pattern)
    :type _any_: MultiFieldPatternData
    :param input.view.entries.dataset: List of input dataset IDs
    :type input.view.entries.dataset: Sequence[str]
    :param input.view.entries.version: List of input dataset version IDs
    :type input.view.entries.version: Sequence[str]
    """

    _service = "tasks"
    _action = "get_all"
    _version = "1.9"
    _schema = {
        'definitions': {
            'multi_field_pattern_data': {
                'properties': {
                    'fields': {
                        'description': 'List of field names',
                        'items': {'type': 'string'},
                        'type': ['array', 'null'],
                    },
                    'pattern': {
                        'description': 'Pattern string (regex)',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'task_status_enum': {
                'enum': [
                    'created',
                    'queued',
                    'in_progress',
                    'stopped',
                    'published',
                    'publishing',
                    'closed',
                    'failed',
                    'unknown',
                ],
                'type': 'string',
            },
        },
        'dependencies': {'page': ['page_size']},
        'properties': {
            '_all_': {
                'description': 'Multi-field pattern condition (all fields match pattern)',
                'oneOf': [
                    {'$ref': '#/definitions/multi_field_pattern_data'},
                    {'type': 'null'},
                ],
            },
            '_any_': {
                'description': 'Multi-field pattern condition (any field matches pattern)',
                'oneOf': [
                    {'$ref': '#/definitions/multi_field_pattern_data'},
                    {'type': 'null'},
                ],
            },
            'id': {
                'description': 'List of IDs to filter by',
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
            'input.view.entries.dataset': {
                'description': 'List of input dataset IDs',
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
            'input.view.entries.version': {
                'description': 'List of input dataset version IDs',
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
            'name': {
                'description': 'Get only tasks whose name matches this pattern (python regular expression syntax)',
                'type': ['string', 'null'],
            },
            'only_fields': {
                'description': "List of task field names (nesting is supported using '.', e.g. execution.model_labels). If provided, this list defines the query's projection (only these fields will be returned for each result entry)",
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
            'order_by': {
                'description': "List of field names to order by. When search_text is used, '@text_score' can be used as a field representing the text score of returned documents. Use '-' prefix to specify descending order. Optional, recommended when using page",
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
            'page': {
                'description': 'Page number, returns a specific page out of the resulting list of tasks',
                'minimum': 0,
                'type': ['integer', 'null'],
            },
            'page_size': {
                'description': 'Page size, specifies the number of results returned in each page (last page may contain fewer results)',
                'minimum': 1,
                'type': ['integer', 'null'],
            },
            'parent': {'description': 'Parent ID', 'type': ['string', 'null']},
            'project': {
                'description': 'List of project IDs',
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
            'search_text': {
                'description': 'Free text search query',
                'type': ['string', 'null'],
            },
            'status': {
                'description': 'List of task status.',
                'items': {'$ref': '#/definitions/task_status_enum'},
                'type': ['array', 'null'],
            },
            'status_changed': {
                'description': 'List of status changed constraint strings (utcformat, epoch) with an optional prefix modifier (>, >=, <, <=)',
                'items': {'pattern': '^(>=|>|<=|<)?.*$', 'type': 'string'},
                'type': ['array', 'null'],
            },
            'tags': {
                'description': "List of task tags. Use '-' prefix to exclude tags",
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
            'type': {
                'description': "List of task types. One or more of: 'import', 'annotation', 'training' or 'testing' (case insensitive)",
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
            'user': {
                'description': "List of user IDs used to filter results by the task's creating user",
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, id=None, name=None, user=None, project=None, page=None, page_size=None, order_by=None, type=None, tags=None, status=None, only_fields=None, parent=None, status_changed=None, search_text=None, _all_=None, _any_=None, input__view__entries__dataset=None, input__view__entries__version=None, **kwargs):
        super(GetAllRequest, self).__init__(**kwargs)
        self.id = id
        self.name = name
        self.user = user
        self.project = project
        self.page = page
        self.page_size = page_size
        self.order_by = order_by
        self.type = type
        self.tags = tags
        self.status = status
        self.only_fields = only_fields
        self.parent = parent
        self.status_changed = status_changed
        self.search_text = search_text
        self._all_ = _all_
        self._any_ = _any_
        self.input__view__entries__dataset = input__view__entries__dataset
        self.input__view__entries__version = input__view__entries__version

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

    @schema_property('user')
    def user(self):
        return self._property_user

    @user.setter
    def user(self, value):
        if value is None:
            self._property_user = None
            return
        
        self.assert_isinstance(value, "user", (list, tuple))
        
        self.assert_isinstance(value, "user", six.string_types, is_array=True)
        self._property_user = value

    @schema_property('project')
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return
        
        self.assert_isinstance(value, "project", (list, tuple))
        
        self.assert_isinstance(value, "project", six.string_types, is_array=True)
        self._property_project = value

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

    @schema_property('type')
    def type(self):
        return self._property_type

    @type.setter
    def type(self, value):
        if value is None:
            self._property_type = None
            return
        
        self.assert_isinstance(value, "type", (list, tuple))
        
        self.assert_isinstance(value, "type", six.string_types, is_array=True)
        self._property_type = value

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

    @schema_property('status')
    def status(self):
        return self._property_status

    @status.setter
    def status(self, value):
        if value is None:
            self._property_status = None
            return
        
        self.assert_isinstance(value, "status", (list, tuple))
        if any(isinstance(v, six.string_types) for v in value):
            value = [TaskStatusEnum(v) if isinstance(v, six.string_types) else v for v in value]
        else:
            self.assert_isinstance(value, "status", TaskStatusEnum, is_array=True)
        self._property_status = value

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

    @schema_property('parent')
    def parent(self):
        return self._property_parent

    @parent.setter
    def parent(self, value):
        if value is None:
            self._property_parent = None
            return
        
        self.assert_isinstance(value, "parent", six.string_types)
        self._property_parent = value

    @schema_property('status_changed')
    def status_changed(self):
        return self._property_status_changed

    @status_changed.setter
    def status_changed(self, value):
        if value is None:
            self._property_status_changed = None
            return
        
        self.assert_isinstance(value, "status_changed", (list, tuple))
        
        self.assert_isinstance(value, "status_changed", six.string_types, is_array=True)
        self._property_status_changed = value

    @schema_property('search_text')
    def search_text(self):
        return self._property_search_text

    @search_text.setter
    def search_text(self, value):
        if value is None:
            self._property_search_text = None
            return
        
        self.assert_isinstance(value, "search_text", six.string_types)
        self._property_search_text = value

    @schema_property('_all_')
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

    @schema_property('_any_')
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

    @schema_property('input.view.entries.dataset')
    def input__view__entries__dataset(self):
        return self._property_input__view__entries__dataset

    @input__view__entries__dataset.setter
    def input__view__entries__dataset(self, value):
        if value is None:
            self._property_input__view__entries__dataset = None
            return
        
        self.assert_isinstance(value, "input__view__entries__dataset", (list, tuple))
        
        self.assert_isinstance(value, "input__view__entries__dataset", six.string_types, is_array=True)
        self._property_input__view__entries__dataset = value

    @schema_property('input.view.entries.version')
    def input__view__entries__version(self):
        return self._property_input__view__entries__version

    @input__view__entries__version.setter
    def input__view__entries__version(self, value):
        if value is None:
            self._property_input__view__entries__version = None
            return
        
        self.assert_isinstance(value, "input__view__entries__version", (list, tuple))
        
        self.assert_isinstance(value, "input__view__entries__version", six.string_types, is_array=True)
        self._property_input__view__entries__version = value


class GetAllResponse(Response):
    """
    Response of tasks.get_all endpoint.

    :param tasks: List of tasks
    :type tasks: Sequence[Task]
    """
    _service = "tasks"
    _action = "get_all"
    _version = "1.9"

    _schema = {
        'definitions': {
            'augmentation': {
                'properties': {
                    'crop_around_rois': {
                        'description': 'Crop image data around all frame ROIs',
                        'type': ['boolean', 'null'],
                    },
                    'sets': {
                        'description': 'List of augmentation sets',
                        'items': {'$ref': '#/definitions/augmentation_set'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'augmentation_set': {
                'properties': {
                    'arguments': {
                        'additionalProperties': {
                            'additionalProperties': True,
                            'type': 'object',
                        },
                        'description': 'Arguments dictionary per custom augmentation type.',
                        'type': ['object', 'null'],
                    },
                    'cls': {
                        'description': 'Augmentation class',
                        'type': ['string', 'null'],
                    },
                    'strength': {
                        'description': 'Augmentation strength. Range [0,).',
                        'minimum': 0,
                        'type': ['number', 'null'],
                    },
                    'types': {
                        'description': 'Augmentation type',
                        'items': {'type': 'string'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'execution': {
                'properties': {
                    'dataviews': {
                        'description': 'Additional dataviews for the task',
                        'items': {'additionalProperties': True, 'type': 'object'},
                        'type': ['array', 'null'],
                    },
                    'framework': {
                        'description': 'Framework related to the task. Case insensitive. Mandatory for Training tasks. ',
                        'type': ['string', 'null'],
                    },
                    'model': {
                        'description': 'Execution input model ID Not applicable for Register (Import) tasks',
                        'type': ['string', 'null'],
                    },
                    'model_desc': {
                        'additionalProperties': True,
                        'description': 'Json object representing the Model descriptors',
                        'type': ['object', 'null'],
                    },
                    'model_labels': {
                        'additionalProperties': {'type': 'integer'},
                        'description': "Json object representing the ids of the labels in the model.\n                The keys are the layers' names and the values are the IDs.\n                Not applicable for Register (Import) tasks.\n                Mandatory for Training tasks[z]",
                        'type': ['object', 'null'],
                    },
                    'parameters': {
                        'additionalProperties': True,
                        'description': 'Json object containing the Task parameters',
                        'type': ['object', 'null'],
                    },
                    'queue': {
                        'description': 'Queue ID where task was queued.',
                        'type': ['string', 'null'],
                    },
                    'test_split': {
                        'description': 'Percentage of frames to use for testing only',
                        'type': ['integer', 'null'],
                    },
                },
                'type': 'object',
            },
            'filter_by_roi_enum': {
                'default': 'label_rules',
                'enum': ['disabled', 'no_rois', 'label_rules'],
                'type': 'string',
            },
            'filter_label_rule': {
                'properties': {
                    'conf_range': {
                        'description': 'Range of ROI confidence level in the frame (min, max). -1 for not applicable\n            Both min and max can be either -1 or positive.\n            2nd number (max) must be either -1 or larger than or equal to the 1st number (min)',
                        'items': {'type': 'number'},
                        'maxItems': 2,
                        'minItems': 1,
                        'type': 'array',
                    },
                    'count_range': {
                        'description': 'Range of times ROI appears in the frame (min, max). -1 for not applicable.\n            Both integers must be larger than or equal to -1.\n            2nd integer (max) must be either -1 or larger than or equal to the 1st integer (min)',
                        'items': {'type': 'integer'},
                        'maxItems': 2,
                        'minItems': 1,
                        'type': 'array',
                    },
                    'label': {
                        'description': "Lucene format query (see lucene query syntax).\nDefault search field is label.keyword and default operator is AND, so searching for:\n\n'Bus Stop' Blue\n\nis equivalent to:\n\nLabel.keyword:'Bus Stop' AND label.keyword:'Blue'",
                        'type': 'string',
                    },
                },
                'required': ['label'],
                'type': 'object',
            },
            'filter_rule': {
                'properties': {
                    'dataset': {
                        'description': "Dataset ID. Must be a dataset which is in the task's view. If set to '*' all datasets in View are used.",
                        'type': 'string',
                    },
                    'filter_by_roi': {
                        '$ref': '#/definitions/filter_by_roi_enum',
                        'description': 'Type of filter',
                    },
                    'frame_query': {
                        'description': 'Frame filter, in Lucene query syntax',
                        'type': 'string',
                    },
                    'label_rules': {
                        'description': "List of FilterLabelRule ('AND' connection)\n\ndisabled - No filtering by ROIs. Select all frames, even if they don't have ROIs (all frames)\n\nno_rois - Select only frames without ROIs (empty frames)\n\nlabel_rules - Select frames according to label rules",
                        'items': {'$ref': '#/definitions/filter_label_rule'},
                        'type': ['array', 'null'],
                    },
                    'sources_query': {
                        'description': 'Sources filter, in Lucene query syntax. Filters sources in each frame.',
                        'type': 'string',
                    },
                    'version': {
                        'description': "Dataset version to apply rule to. Must belong to the dataset and be in the task's view. If set to '*' all version of the datasets in View are used.",
                        'type': 'string',
                    },
                    'weight': {
                        'description': 'Rule weight. Default is 1',
                        'type': 'number',
                    },
                },
                'required': ['filter_by_roi'],
                'type': 'object',
            },
            'filtering': {
                'properties': {
                    'filtering_rules': {
                        'description': "List of FilterRule ('OR' connection)",
                        'items': {'$ref': '#/definitions/filter_rule'},
                        'type': ['array', 'null'],
                    },
                    'output_rois': {
                        'description': "'all_in_frame' - all rois for a frame are returned\n\n'only_filtered' - only rois which led this frame to be selected\n\n'frame_per_roi' - single roi per frame. Frame can be returned multiple times with a different roi each time.\n\nNote: this should be used for Training tasks only\n\nNote: frame_per_roi implies that only filtered rois will be returned\n                ",
                        'oneOf': [
                            {'$ref': '#/definitions/output_rois_enum'},
                            {'type': 'null'},
                        ],
                    },
                },
                'type': 'object',
            },
            'input': {
                'properties': {
                    'augmentation': {
                        'description': 'Augmentation parameters. Only for training and testing tasks.',
                        'oneOf': [
                            {'$ref': '#/definitions/augmentation'},
                            {'type': 'null'},
                        ],
                    },
                    'dataviews': {
                        'additionalProperties': {'type': 'string'},
                        'description': 'Key to DataView ID Mapping',
                        'type': ['object', 'null'],
                    },
                    'frames_filter': {
                        'description': 'Filtering params',
                        'oneOf': [
                            {'$ref': '#/definitions/filtering'},
                            {'type': 'null'},
                        ],
                    },
                    'iteration': {
                        'description': 'Iteration parameters. Not applicable for register (import) tasks.',
                        'oneOf': [
                            {'$ref': '#/definitions/iteration'},
                            {'type': 'null'},
                        ],
                    },
                    'mapping': {
                        'description': 'Mapping params (see common definitions section)',
                        'oneOf': [
                            {'$ref': '#/definitions/mapping'},
                            {'type': 'null'},
                        ],
                    },
                    'view': {
                        'description': 'View params',
                        'oneOf': [{'$ref': '#/definitions/view'}, {'type': 'null'}],
                    },
                },
                'type': 'object',
            },
            'iteration': {
                'description': 'Sequential Iteration API configuration',
                'properties': {
                    'infinite': {
                        'description': 'Infinite iteration',
                        'type': ['boolean', 'null'],
                    },
                    'jump': {
                        'description': 'Jump entry',
                        'oneOf': [{'$ref': '#/definitions/jump'}, {'type': 'null'}],
                    },
                    'limit': {
                        'description': 'Maximum frames per task. If not passed, frames will end when no more matching frames are found, unless infinite is True.',
                        'type': ['integer', 'null'],
                    },
                    'min_sequence': {
                        'description': 'Length (in ms) of video clips to return. This is used in random order, and in sequential order only if jumping is provided and only for video frames',
                        'type': ['integer', 'null'],
                    },
                    'order': {
                        'description': "\n                Input frames order. Values: 'sequential', 'random'\n                In Sequential mode frames will be returned according to the order in which the frames were added to the dataset.",
                        'type': ['string', 'null'],
                    },
                    'random_seed': {
                        'description': 'Random seed used during iteration',
                        'type': 'integer',
                    },
                },
                'required': ['random_seed'],
                'type': 'object',
            },
            'jump': {
                'properties': {
                    'time': {
                        'description': 'Max time in milliseconds between frames',
                        'type': ['integer', 'null'],
                    },
                },
                'type': 'object',
            },
            'label_source': {
                'properties': {
                    'dataset': {
                        'description': "Source dataset id. '*' for all datasets in view",
                        'type': ['string', 'null'],
                    },
                    'labels': {
                        'description': "List of source labels (AND connection). '*' indicates any label. Labels must exist in at least one of the dataset versions in the task's view",
                        'items': {'type': 'string'},
                        'type': ['array', 'null'],
                    },
                    'version': {
                        'description': "Source dataset version id. Default is '*' (for all versions in dataset in the view) Version must belong to the selected dataset, and must be in the task's view[i]",
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'last_metrics_event': {
                'properties': {
                    'iter': {
                        'description': 'Iteration number',
                        'type': ['integer', 'null'],
                    },
                    'metric': {
                        'description': 'Metric name',
                        'type': ['string', 'null'],
                    },
                    'timestamp': {
                        'description': 'Event report time (UTC)',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'type': {
                        'description': 'Event type',
                        'type': ['string', 'null'],
                    },
                    'value': {'description': 'Value', 'type': ['number', 'null']},
                    'variant': {
                        'description': 'Variant name',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'last_metrics_variants': {
                'additionalProperties': {
                    '$ref': '#/definitions/last_metrics_event',
                },
                'description': 'Last metric events, one for each variant hash',
                'type': 'object',
            },
            'mapping': {
                'properties': {
                    'rules': {
                        'description': 'Rules list',
                        'items': {'$ref': '#/definitions/mapping_rule'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'mapping_rule': {
                'properties': {
                    'source': {
                        'description': 'Source label info',
                        'oneOf': [
                            {'$ref': '#/definitions/label_source'},
                            {'type': 'null'},
                        ],
                    },
                    'target': {
                        'description': 'Target label name',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'output': {
                'properties': {
                    'destination': {
                        'description': 'Storage id. This is where output files will be stored.',
                        'type': ['string', 'null'],
                    },
                    'error': {
                        'description': 'Last error text',
                        'type': ['string', 'null'],
                    },
                    'model': {
                        'description': 'Model id.',
                        'type': ['string', 'null'],
                    },
                    'result': {
                        'description': "Task result. Values: 'success', 'failure'",
                        'type': ['string', 'null'],
                    },
                    'view': {
                        'description': 'View params',
                        'oneOf': [{'$ref': '#/definitions/view'}, {'type': 'null'}],
                    },
                },
                'type': 'object',
            },
            'output_rois_enum': {
                'enum': ['all_in_frame', 'only_filtered', 'frame_per_roi'],
                'type': 'string',
            },
            'script': {
                'properties': {
                    'binary': {
                        'default': 'python',
                        'description': 'Binary to use when running the script',
                        'type': ['string', 'null'],
                    },
                    'branch': {
                        'description': 'Repository branch id If not provided and tag not provided, default repository branch is used.',
                        'type': ['string', 'null'],
                    },
                    'entry_point': {
                        'description': 'Path to execute within the repository',
                        'type': ['string', 'null'],
                    },
                    'repository': {
                        'description': 'Name of the repository where the script is located',
                        'type': ['string', 'null'],
                    },
                    'requirements': {
                        'description': 'A JSON object containing requirements strings by key',
                        'type': ['object', 'null'],
                    },
                    'tag': {
                        'description': 'Repository tag',
                        'type': ['string', 'null'],
                    },
                    'version_num': {
                        'description': 'Version (changeset) number. Optional (default is head version) Unused if tag is provided.',
                        'type': ['string', 'null'],
                    },
                    'working_dir': {
                        'description': 'Path to the folder from which to run the script Default - root folder of repository[f]',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'task': {
                'properties': {
                    'comment': {
                        'description': 'Free text comment',
                        'type': ['string', 'null'],
                    },
                    'company': {
                        'description': 'Company ID',
                        'type': ['string', 'null'],
                    },
                    'completed': {
                        'description': 'Task end time (UTC)',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'created': {
                        'description': 'Task creation time (UTC) ',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'execution': {
                        'description': 'Task execution params',
                        'oneOf': [
                            {'$ref': '#/definitions/execution'},
                            {'type': 'null'},
                        ],
                    },
                    'id': {'description': 'Task id', 'type': ['string', 'null']},
                    'input': {
                        'description': 'Task input params',
                        'oneOf': [
                            {'$ref': '#/definitions/input'},
                            {'type': 'null'},
                        ],
                    },
                    'last_iteration': {
                        'description': 'Last iteration reported for this task',
                        'type': ['integer', 'null'],
                    },
                    'last_metrics': {
                        'additionalProperties': {
                            '$ref': '#/definitions/last_metrics_variants',
                        },
                        'description': 'Last metric variants (hash to events), one for each metric hash',
                        'type': ['object', 'null'],
                    },
                    'last_update': {
                        'description': 'Last time this task was created, updated, changed or events for this task were reported',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'last_worker': {
                        'description': 'ID of last worker that handled the task',
                        'type': ['string', 'null'],
                    },
                    'last_worker_report': {
                        'description': 'Last time a worker reported while working on this task',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'name': {
                        'description': 'Task Name',
                        'type': ['string', 'null'],
                    },
                    'output': {
                        'description': 'Task output params',
                        'oneOf': [
                            {'$ref': '#/definitions/output'},
                            {'type': 'null'},
                        ],
                    },
                    'parent': {
                        'description': 'Parent task id',
                        'type': ['string', 'null'],
                    },
                    'project': {
                        'description': 'Project ID of the project to which this task is assigned',
                        'type': ['string', 'null'],
                    },
                    'published': {
                        'description': 'Last status change time',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'script': {
                        'description': 'Script info',
                        'oneOf': [
                            {'$ref': '#/definitions/script'},
                            {'type': 'null'},
                        ],
                    },
                    'started': {
                        'description': 'Task start time (UTC)',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'status': {
                        'description': '',
                        'oneOf': [
                            {'$ref': '#/definitions/task_status_enum'},
                            {'type': 'null'},
                        ],
                    },
                    'status_changed': {
                        'description': 'Last status change time',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'status_message': {
                        'description': 'free text string representing info about the status',
                        'type': ['string', 'null'],
                    },
                    'status_reason': {
                        'description': 'Reason for last status change',
                        'type': ['string', 'null'],
                    },
                    'tags': {
                        'description': 'Tags list',
                        'items': {'type': 'string'},
                        'type': ['array', 'null'],
                    },
                    'type': {
                        'description': "Type of task. Values: 'dataset_import', 'annotation', 'training', 'testing'",
                        'oneOf': [
                            {'$ref': '#/definitions/task_type_enum'},
                            {'type': 'null'},
                        ],
                    },
                    'user': {
                        'description': 'Associated user id',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'task_status_enum': {
                'enum': [
                    'created',
                    'queued',
                    'in_progress',
                    'stopped',
                    'published',
                    'publishing',
                    'closed',
                    'failed',
                    'unknown',
                ],
                'type': 'string',
            },
            'task_type_enum': {
                'enum': [
                    'dataset_import',
                    'annotation',
                    'annotation_manual',
                    'training',
                    'testing',
                ],
                'type': 'string',
            },
            'view': {
                'properties': {
                    'entries': {
                        'description': 'List of view entries. All tasks must have at least one view.',
                        'items': {'$ref': '#/definitions/view_entry'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'view_entry': {
                'properties': {
                    'dataset': {
                        'description': 'Existing Dataset id',
                        'type': ['string', 'null'],
                    },
                    'merge_with': {
                        'description': 'Version ID to merge with',
                        'type': ['string', 'null'],
                    },
                    'version': {
                        'description': 'Version id of a version belonging to the dataset',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
        },
        'properties': {
            'tasks': {
                'description': 'List of tasks',
                'items': {'$ref': '#/definitions/task'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, tasks=None, **kwargs):
        super(GetAllResponse, self).__init__(**kwargs)
        self.tasks = tasks

    @schema_property('tasks')
    def tasks(self):
        return self._property_tasks

    @tasks.setter
    def tasks(self, value):
        if value is None:
            self._property_tasks = None
            return
        
        self.assert_isinstance(value, "tasks", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [Task.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "tasks", Task, is_array=True)
        self._property_tasks = value


class GetByIdRequest(Request):
    """
    Gets task information

    :param task: Task ID
    :type task: str
    """

    _service = "tasks"
    _action = "get_by_id"
    _version = "1.9"
    _schema = {
        'definitions': {},
        'properties': {'task': {'description': 'Task ID', 'type': 'string'}},
        'required': ['task'],
        'type': 'object',
    }
    def __init__(
            self, task, **kwargs):
        super(GetByIdRequest, self).__init__(**kwargs)
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


class GetByIdResponse(Response):
    """
    Response of tasks.get_by_id endpoint.

    :param task: Task info
    :type task: Task
    """
    _service = "tasks"
    _action = "get_by_id"
    _version = "1.9"

    _schema = {
        'definitions': {
            'augmentation': {
                'properties': {
                    'crop_around_rois': {
                        'description': 'Crop image data around all frame ROIs',
                        'type': ['boolean', 'null'],
                    },
                    'sets': {
                        'description': 'List of augmentation sets',
                        'items': {'$ref': '#/definitions/augmentation_set'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'augmentation_set': {
                'properties': {
                    'arguments': {
                        'additionalProperties': {
                            'additionalProperties': True,
                            'type': 'object',
                        },
                        'description': 'Arguments dictionary per custom augmentation type.',
                        'type': ['object', 'null'],
                    },
                    'cls': {
                        'description': 'Augmentation class',
                        'type': ['string', 'null'],
                    },
                    'strength': {
                        'description': 'Augmentation strength. Range [0,).',
                        'minimum': 0,
                        'type': ['number', 'null'],
                    },
                    'types': {
                        'description': 'Augmentation type',
                        'items': {'type': 'string'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'execution': {
                'properties': {
                    'dataviews': {
                        'description': 'Additional dataviews for the task',
                        'items': {'additionalProperties': True, 'type': 'object'},
                        'type': ['array', 'null'],
                    },
                    'framework': {
                        'description': 'Framework related to the task. Case insensitive. Mandatory for Training tasks. ',
                        'type': ['string', 'null'],
                    },
                    'model': {
                        'description': 'Execution input model ID Not applicable for Register (Import) tasks',
                        'type': ['string', 'null'],
                    },
                    'model_desc': {
                        'additionalProperties': True,
                        'description': 'Json object representing the Model descriptors',
                        'type': ['object', 'null'],
                    },
                    'model_labels': {
                        'additionalProperties': {'type': 'integer'},
                        'description': "Json object representing the ids of the labels in the model.\n                The keys are the layers' names and the values are the IDs.\n                Not applicable for Register (Import) tasks.\n                Mandatory for Training tasks[z]",
                        'type': ['object', 'null'],
                    },
                    'parameters': {
                        'additionalProperties': True,
                        'description': 'Json object containing the Task parameters',
                        'type': ['object', 'null'],
                    },
                    'queue': {
                        'description': 'Queue ID where task was queued.',
                        'type': ['string', 'null'],
                    },
                    'test_split': {
                        'description': 'Percentage of frames to use for testing only',
                        'type': ['integer', 'null'],
                    },
                },
                'type': 'object',
            },
            'filter_by_roi_enum': {
                'default': 'label_rules',
                'enum': ['disabled', 'no_rois', 'label_rules'],
                'type': 'string',
            },
            'filter_label_rule': {
                'properties': {
                    'conf_range': {
                        'description': 'Range of ROI confidence level in the frame (min, max). -1 for not applicable\n            Both min and max can be either -1 or positive.\n            2nd number (max) must be either -1 or larger than or equal to the 1st number (min)',
                        'items': {'type': 'number'},
                        'maxItems': 2,
                        'minItems': 1,
                        'type': 'array',
                    },
                    'count_range': {
                        'description': 'Range of times ROI appears in the frame (min, max). -1 for not applicable.\n            Both integers must be larger than or equal to -1.\n            2nd integer (max) must be either -1 or larger than or equal to the 1st integer (min)',
                        'items': {'type': 'integer'},
                        'maxItems': 2,
                        'minItems': 1,
                        'type': 'array',
                    },
                    'label': {
                        'description': "Lucene format query (see lucene query syntax).\nDefault search field is label.keyword and default operator is AND, so searching for:\n\n'Bus Stop' Blue\n\nis equivalent to:\n\nLabel.keyword:'Bus Stop' AND label.keyword:'Blue'",
                        'type': 'string',
                    },
                },
                'required': ['label'],
                'type': 'object',
            },
            'filter_rule': {
                'properties': {
                    'dataset': {
                        'description': "Dataset ID. Must be a dataset which is in the task's view. If set to '*' all datasets in View are used.",
                        'type': 'string',
                    },
                    'filter_by_roi': {
                        '$ref': '#/definitions/filter_by_roi_enum',
                        'description': 'Type of filter',
                    },
                    'frame_query': {
                        'description': 'Frame filter, in Lucene query syntax',
                        'type': 'string',
                    },
                    'label_rules': {
                        'description': "List of FilterLabelRule ('AND' connection)\n\ndisabled - No filtering by ROIs. Select all frames, even if they don't have ROIs (all frames)\n\nno_rois - Select only frames without ROIs (empty frames)\n\nlabel_rules - Select frames according to label rules",
                        'items': {'$ref': '#/definitions/filter_label_rule'},
                        'type': ['array', 'null'],
                    },
                    'sources_query': {
                        'description': 'Sources filter, in Lucene query syntax. Filters sources in each frame.',
                        'type': 'string',
                    },
                    'version': {
                        'description': "Dataset version to apply rule to. Must belong to the dataset and be in the task's view. If set to '*' all version of the datasets in View are used.",
                        'type': 'string',
                    },
                    'weight': {
                        'description': 'Rule weight. Default is 1',
                        'type': 'number',
                    },
                },
                'required': ['filter_by_roi'],
                'type': 'object',
            },
            'filtering': {
                'properties': {
                    'filtering_rules': {
                        'description': "List of FilterRule ('OR' connection)",
                        'items': {'$ref': '#/definitions/filter_rule'},
                        'type': ['array', 'null'],
                    },
                    'output_rois': {
                        'description': "'all_in_frame' - all rois for a frame are returned\n\n'only_filtered' - only rois which led this frame to be selected\n\n'frame_per_roi' - single roi per frame. Frame can be returned multiple times with a different roi each time.\n\nNote: this should be used for Training tasks only\n\nNote: frame_per_roi implies that only filtered rois will be returned\n                ",
                        'oneOf': [
                            {'$ref': '#/definitions/output_rois_enum'},
                            {'type': 'null'},
                        ],
                    },
                },
                'type': 'object',
            },
            'input': {
                'properties': {
                    'augmentation': {
                        'description': 'Augmentation parameters. Only for training and testing tasks.',
                        'oneOf': [
                            {'$ref': '#/definitions/augmentation'},
                            {'type': 'null'},
                        ],
                    },
                    'dataviews': {
                        'additionalProperties': {'type': 'string'},
                        'description': 'Key to DataView ID Mapping',
                        'type': ['object', 'null'],
                    },
                    'frames_filter': {
                        'description': 'Filtering params',
                        'oneOf': [
                            {'$ref': '#/definitions/filtering'},
                            {'type': 'null'},
                        ],
                    },
                    'iteration': {
                        'description': 'Iteration parameters. Not applicable for register (import) tasks.',
                        'oneOf': [
                            {'$ref': '#/definitions/iteration'},
                            {'type': 'null'},
                        ],
                    },
                    'mapping': {
                        'description': 'Mapping params (see common definitions section)',
                        'oneOf': [
                            {'$ref': '#/definitions/mapping'},
                            {'type': 'null'},
                        ],
                    },
                    'view': {
                        'description': 'View params',
                        'oneOf': [{'$ref': '#/definitions/view'}, {'type': 'null'}],
                    },
                },
                'type': 'object',
            },
            'iteration': {
                'description': 'Sequential Iteration API configuration',
                'properties': {
                    'infinite': {
                        'description': 'Infinite iteration',
                        'type': ['boolean', 'null'],
                    },
                    'jump': {
                        'description': 'Jump entry',
                        'oneOf': [{'$ref': '#/definitions/jump'}, {'type': 'null'}],
                    },
                    'limit': {
                        'description': 'Maximum frames per task. If not passed, frames will end when no more matching frames are found, unless infinite is True.',
                        'type': ['integer', 'null'],
                    },
                    'min_sequence': {
                        'description': 'Length (in ms) of video clips to return. This is used in random order, and in sequential order only if jumping is provided and only for video frames',
                        'type': ['integer', 'null'],
                    },
                    'order': {
                        'description': "\n                Input frames order. Values: 'sequential', 'random'\n                In Sequential mode frames will be returned according to the order in which the frames were added to the dataset.",
                        'type': ['string', 'null'],
                    },
                    'random_seed': {
                        'description': 'Random seed used during iteration',
                        'type': 'integer',
                    },
                },
                'required': ['random_seed'],
                'type': 'object',
            },
            'jump': {
                'properties': {
                    'time': {
                        'description': 'Max time in milliseconds between frames',
                        'type': ['integer', 'null'],
                    },
                },
                'type': 'object',
            },
            'label_source': {
                'properties': {
                    'dataset': {
                        'description': "Source dataset id. '*' for all datasets in view",
                        'type': ['string', 'null'],
                    },
                    'labels': {
                        'description': "List of source labels (AND connection). '*' indicates any label. Labels must exist in at least one of the dataset versions in the task's view",
                        'items': {'type': 'string'},
                        'type': ['array', 'null'],
                    },
                    'version': {
                        'description': "Source dataset version id. Default is '*' (for all versions in dataset in the view) Version must belong to the selected dataset, and must be in the task's view[i]",
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'last_metrics_event': {
                'properties': {
                    'iter': {
                        'description': 'Iteration number',
                        'type': ['integer', 'null'],
                    },
                    'metric': {
                        'description': 'Metric name',
                        'type': ['string', 'null'],
                    },
                    'timestamp': {
                        'description': 'Event report time (UTC)',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'type': {
                        'description': 'Event type',
                        'type': ['string', 'null'],
                    },
                    'value': {'description': 'Value', 'type': ['number', 'null']},
                    'variant': {
                        'description': 'Variant name',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'last_metrics_variants': {
                'additionalProperties': {
                    '$ref': '#/definitions/last_metrics_event',
                },
                'description': 'Last metric events, one for each variant hash',
                'type': 'object',
            },
            'mapping': {
                'properties': {
                    'rules': {
                        'description': 'Rules list',
                        'items': {'$ref': '#/definitions/mapping_rule'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'mapping_rule': {
                'properties': {
                    'source': {
                        'description': 'Source label info',
                        'oneOf': [
                            {'$ref': '#/definitions/label_source'},
                            {'type': 'null'},
                        ],
                    },
                    'target': {
                        'description': 'Target label name',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'output': {
                'properties': {
                    'destination': {
                        'description': 'Storage id. This is where output files will be stored.',
                        'type': ['string', 'null'],
                    },
                    'error': {
                        'description': 'Last error text',
                        'type': ['string', 'null'],
                    },
                    'model': {
                        'description': 'Model id.',
                        'type': ['string', 'null'],
                    },
                    'result': {
                        'description': "Task result. Values: 'success', 'failure'",
                        'type': ['string', 'null'],
                    },
                    'view': {
                        'description': 'View params',
                        'oneOf': [{'$ref': '#/definitions/view'}, {'type': 'null'}],
                    },
                },
                'type': 'object',
            },
            'output_rois_enum': {
                'enum': ['all_in_frame', 'only_filtered', 'frame_per_roi'],
                'type': 'string',
            },
            'script': {
                'properties': {
                    'binary': {
                        'default': 'python',
                        'description': 'Binary to use when running the script',
                        'type': ['string', 'null'],
                    },
                    'branch': {
                        'description': 'Repository branch id If not provided and tag not provided, default repository branch is used.',
                        'type': ['string', 'null'],
                    },
                    'entry_point': {
                        'description': 'Path to execute within the repository',
                        'type': ['string', 'null'],
                    },
                    'repository': {
                        'description': 'Name of the repository where the script is located',
                        'type': ['string', 'null'],
                    },
                    'requirements': {
                        'description': 'A JSON object containing requirements strings by key',
                        'type': ['object', 'null'],
                    },
                    'tag': {
                        'description': 'Repository tag',
                        'type': ['string', 'null'],
                    },
                    'version_num': {
                        'description': 'Version (changeset) number. Optional (default is head version) Unused if tag is provided.',
                        'type': ['string', 'null'],
                    },
                    'working_dir': {
                        'description': 'Path to the folder from which to run the script Default - root folder of repository[f]',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'task': {
                'properties': {
                    'comment': {
                        'description': 'Free text comment',
                        'type': ['string', 'null'],
                    },
                    'company': {
                        'description': 'Company ID',
                        'type': ['string', 'null'],
                    },
                    'completed': {
                        'description': 'Task end time (UTC)',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'created': {
                        'description': 'Task creation time (UTC) ',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'execution': {
                        'description': 'Task execution params',
                        'oneOf': [
                            {'$ref': '#/definitions/execution'},
                            {'type': 'null'},
                        ],
                    },
                    'id': {'description': 'Task id', 'type': ['string', 'null']},
                    'input': {
                        'description': 'Task input params',
                        'oneOf': [
                            {'$ref': '#/definitions/input'},
                            {'type': 'null'},
                        ],
                    },
                    'last_iteration': {
                        'description': 'Last iteration reported for this task',
                        'type': ['integer', 'null'],
                    },
                    'last_metrics': {
                        'additionalProperties': {
                            '$ref': '#/definitions/last_metrics_variants',
                        },
                        'description': 'Last metric variants (hash to events), one for each metric hash',
                        'type': ['object', 'null'],
                    },
                    'last_update': {
                        'description': 'Last time this task was created, updated, changed or events for this task were reported',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'last_worker': {
                        'description': 'ID of last worker that handled the task',
                        'type': ['string', 'null'],
                    },
                    'last_worker_report': {
                        'description': 'Last time a worker reported while working on this task',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'name': {
                        'description': 'Task Name',
                        'type': ['string', 'null'],
                    },
                    'output': {
                        'description': 'Task output params',
                        'oneOf': [
                            {'$ref': '#/definitions/output'},
                            {'type': 'null'},
                        ],
                    },
                    'parent': {
                        'description': 'Parent task id',
                        'type': ['string', 'null'],
                    },
                    'project': {
                        'description': 'Project ID of the project to which this task is assigned',
                        'type': ['string', 'null'],
                    },
                    'published': {
                        'description': 'Last status change time',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'script': {
                        'description': 'Script info',
                        'oneOf': [
                            {'$ref': '#/definitions/script'},
                            {'type': 'null'},
                        ],
                    },
                    'started': {
                        'description': 'Task start time (UTC)',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'status': {
                        'description': '',
                        'oneOf': [
                            {'$ref': '#/definitions/task_status_enum'},
                            {'type': 'null'},
                        ],
                    },
                    'status_changed': {
                        'description': 'Last status change time',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'status_message': {
                        'description': 'free text string representing info about the status',
                        'type': ['string', 'null'],
                    },
                    'status_reason': {
                        'description': 'Reason for last status change',
                        'type': ['string', 'null'],
                    },
                    'tags': {
                        'description': 'Tags list',
                        'items': {'type': 'string'},
                        'type': ['array', 'null'],
                    },
                    'type': {
                        'description': "Type of task. Values: 'dataset_import', 'annotation', 'training', 'testing'",
                        'oneOf': [
                            {'$ref': '#/definitions/task_type_enum'},
                            {'type': 'null'},
                        ],
                    },
                    'user': {
                        'description': 'Associated user id',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'task_status_enum': {
                'enum': [
                    'created',
                    'queued',
                    'in_progress',
                    'stopped',
                    'published',
                    'publishing',
                    'closed',
                    'failed',
                    'unknown',
                ],
                'type': 'string',
            },
            'task_type_enum': {
                'enum': [
                    'dataset_import',
                    'annotation',
                    'annotation_manual',
                    'training',
                    'testing',
                ],
                'type': 'string',
            },
            'view': {
                'properties': {
                    'entries': {
                        'description': 'List of view entries. All tasks must have at least one view.',
                        'items': {'$ref': '#/definitions/view_entry'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'view_entry': {
                'properties': {
                    'dataset': {
                        'description': 'Existing Dataset id',
                        'type': ['string', 'null'],
                    },
                    'merge_with': {
                        'description': 'Version ID to merge with',
                        'type': ['string', 'null'],
                    },
                    'version': {
                        'description': 'Version id of a version belonging to the dataset',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
        },
        'properties': {
            'task': {
                'description': 'Task info',
                'oneOf': [{'$ref': '#/definitions/task'}, {'type': 'null'}],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, task=None, **kwargs):
        super(GetByIdResponse, self).__init__(**kwargs)
        self.task = task

    @schema_property('task')
    def task(self):
        return self._property_task

    @task.setter
    def task(self, value):
        if value is None:
            self._property_task = None
            return
        if isinstance(value, dict):
            value = Task.from_dict(value)
        else:
            self.assert_isinstance(value, "task", Task)
        self._property_task = value


class PublishRequest(Request):
    """
    Mark a task status as published.

            For Annotation tasks - if any changes were committed by this task, a new version in the dataset together with an output view are created.

            For Training tasks - if a model was created, it should be set to ready.

    :param force: If not true, call fails if the task status is not 'stopped'
    :type force: bool
    :param publish_model: Indicates that the task output model (if exists) should
        be published. Optional, the default value is True.
    :type publish_model: bool
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "publish"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'force': {
                'default': False,
                'description': "If not true, call fails if the task status is not 'stopped'",
                'type': ['boolean', 'null'],
            },
            'publish_model': {
                'description': 'Indicates that the task output model (if exists) should be published. Optional, the default value is True.',
                'type': ['boolean', 'null'],
            },
            'status_message': {
                'description': 'Extra information regarding status change',
                'type': 'string',
            },
            'status_reason': {
                'description': 'Reason for status change',
                'type': 'string',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }
    def __init__(
            self, task, force=False, publish_model=None, status_reason=None, status_message=None, **kwargs):
        super(PublishRequest, self).__init__(**kwargs)
        self.force = force
        self.publish_model = publish_model
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property('force')
    def force(self):
        return self._property_force

    @force.setter
    def force(self, value):
        if value is None:
            self._property_force = None
            return
        
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

    @schema_property('publish_model')
    def publish_model(self):
        return self._property_publish_model

    @publish_model.setter
    def publish_model(self, value):
        if value is None:
            self._property_publish_model = None
            return
        
        self.assert_isinstance(value, "publish_model", (bool,))
        self._property_publish_model = value

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

    @schema_property('status_reason')
    def status_reason(self):
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value):
        if value is None:
            self._property_status_reason = None
            return
        
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property('status_message')
    def status_message(self):
        return self._property_status_message

    @status_message.setter
    def status_message(self, value):
        if value is None:
            self._property_status_message = None
            return
        
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class PublishResponse(Response):
    """
    Response of tasks.publish endpoint.

    :param committed_versions_results: Committed versions results
    :type committed_versions_results: Sequence[dict]
    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """
    _service = "tasks"
    _action = "publish"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'committed_versions_results': {
                'description': 'Committed versions results',
                'items': {'additionalProperties': True, 'type': 'object'},
                'type': ['array', 'null'],
            },
            'fields': {
                'additionalProperties': True,
                'description': 'Updated fields names and values',
                'type': ['object', 'null'],
            },
            'updated': {
                'description': 'Number of tasks updated (0 or 1)',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, committed_versions_results=None, updated=None, fields=None, **kwargs):
        super(PublishResponse, self).__init__(**kwargs)
        self.committed_versions_results = committed_versions_results
        self.updated = updated
        self.fields = fields

    @schema_property('committed_versions_results')
    def committed_versions_results(self):
        return self._property_committed_versions_results

    @committed_versions_results.setter
    def committed_versions_results(self, value):
        if value is None:
            self._property_committed_versions_results = None
            return
        
        self.assert_isinstance(value, "committed_versions_results", (list, tuple))
        
        self.assert_isinstance(value, "committed_versions_results", (dict,), is_array=True)
        self._property_committed_versions_results = value

    @schema_property('updated')
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

    @schema_property('fields')
    def fields(self):
        return self._property_fields

    @fields.setter
    def fields(self, value):
        if value is None:
            self._property_fields = None
            return
        
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class ResetRequest(Request):
    """
    Reset a task to its initial state, along with any information stored for it (statistics, frame updates etc.).

    :param force: If not true, call fails if the task status is 'completed'
    :type force: bool
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "reset"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'force': {
                'default': False,
                'description': "If not true, call fails if the task status is 'completed'",
                'type': ['boolean', 'null'],
            },
            'status_message': {
                'description': 'Extra information regarding status change',
                'type': 'string',
            },
            'status_reason': {
                'description': 'Reason for status change',
                'type': 'string',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }
    def __init__(
            self, task, force=False, status_reason=None, status_message=None, **kwargs):
        super(ResetRequest, self).__init__(**kwargs)
        self.force = force
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property('force')
    def force(self):
        return self._property_force

    @force.setter
    def force(self, value):
        if value is None:
            self._property_force = None
            return
        
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

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

    @schema_property('status_reason')
    def status_reason(self):
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value):
        if value is None:
            self._property_status_reason = None
            return
        
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property('status_message')
    def status_message(self):
        return self._property_status_message

    @status_message.setter
    def status_message(self, value):
        if value is None:
            self._property_status_message = None
            return
        
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class ResetResponse(Response):
    """
    Response of tasks.reset endpoint.

    :param deleted_indices: List of deleted ES indices that were removed as part of
        the reset process
    :type deleted_indices: Sequence[str]
    :param dequeued: Response from queues.remove_task
    :type dequeued: dict
    :param frames: Response from frames.rollback
    :type frames: dict
    :param events: Response from events.delete_for_task
    :type events: dict
    :param deleted_models: Number of output models deleted by the reset
    :type deleted_models: int
    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """
    _service = "tasks"
    _action = "reset"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'deleted_indices': {
                'description': 'List of deleted ES indices that were removed as part of the reset process',
                'items': {'type': 'string'},
                'type': ['array', 'null'],
            },
            'deleted_models': {
                'description': 'Number of output models deleted by the reset',
                'type': ['integer', 'null'],
            },
            'dequeued': {
                'additionalProperties': True,
                'description': 'Response from queues.remove_task',
                'type': ['object', 'null'],
            },
            'events': {
                'additionalProperties': True,
                'description': 'Response from events.delete_for_task',
                'type': ['object', 'null'],
            },
            'fields': {
                'additionalProperties': True,
                'description': 'Updated fields names and values',
                'type': ['object', 'null'],
            },
            'frames': {
                'additionalProperties': True,
                'description': 'Response from frames.rollback',
                'type': ['object', 'null'],
            },
            'updated': {
                'description': 'Number of tasks updated (0 or 1)',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, deleted_indices=None, dequeued=None, frames=None, events=None, deleted_models=None, updated=None, fields=None, **kwargs):
        super(ResetResponse, self).__init__(**kwargs)
        self.deleted_indices = deleted_indices
        self.dequeued = dequeued
        self.frames = frames
        self.events = events
        self.deleted_models = deleted_models
        self.updated = updated
        self.fields = fields

    @schema_property('deleted_indices')
    def deleted_indices(self):
        return self._property_deleted_indices

    @deleted_indices.setter
    def deleted_indices(self, value):
        if value is None:
            self._property_deleted_indices = None
            return
        
        self.assert_isinstance(value, "deleted_indices", (list, tuple))
        
        self.assert_isinstance(value, "deleted_indices", six.string_types, is_array=True)
        self._property_deleted_indices = value

    @schema_property('dequeued')
    def dequeued(self):
        return self._property_dequeued

    @dequeued.setter
    def dequeued(self, value):
        if value is None:
            self._property_dequeued = None
            return
        
        self.assert_isinstance(value, "dequeued", (dict,))
        self._property_dequeued = value

    @schema_property('frames')
    def frames(self):
        return self._property_frames

    @frames.setter
    def frames(self, value):
        if value is None:
            self._property_frames = None
            return
        
        self.assert_isinstance(value, "frames", (dict,))
        self._property_frames = value

    @schema_property('events')
    def events(self):
        return self._property_events

    @events.setter
    def events(self, value):
        if value is None:
            self._property_events = None
            return
        
        self.assert_isinstance(value, "events", (dict,))
        self._property_events = value

    @schema_property('deleted_models')
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

    @schema_property('updated')
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

    @schema_property('fields')
    def fields(self):
        return self._property_fields

    @fields.setter
    def fields(self, value):
        if value is None:
            self._property_fields = None
            return
        
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class SetRequirementsRequest(Request):
    """
    Set the script requirements for a task

    :param task: Task ID
    :type task: str
    :param requirements: A JSON object containing requirements strings by key
    :type requirements: dict
    """

    _service = "tasks"
    _action = "set_requirements"
    _version = "1.6"
    _schema = {
        'definitions': {},
        'properties': {
            'requirements': {
                'description': 'A JSON object containing requirements strings by key',
                'type': 'object',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task', 'requirements'],
        'type': 'object',
    }
    def __init__(
            self, task, requirements, **kwargs):
        super(SetRequirementsRequest, self).__init__(**kwargs)
        self.task = task
        self.requirements = requirements

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

    @schema_property('requirements')
    def requirements(self):
        return self._property_requirements

    @requirements.setter
    def requirements(self, value):
        if value is None:
            self._property_requirements = None
            return
        
        self.assert_isinstance(value, "requirements", (dict,))
        self._property_requirements = value


class SetRequirementsResponse(Response):
    """
    Response of tasks.set_requirements endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """
    _service = "tasks"
    _action = "set_requirements"
    _version = "1.6"

    _schema = {
        'definitions': {},
        'properties': {
            'fields': {
                'additionalProperties': True,
                'description': 'Updated fields names and values',
                'type': ['object', 'null'],
            },
            'updated': {
                'description': 'Number of tasks updated (0 or 1)',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, updated=None, fields=None, **kwargs):
        super(SetRequirementsResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property('updated')
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

    @schema_property('fields')
    def fields(self):
        return self._property_fields

    @fields.setter
    def fields(self, value):
        if value is None:
            self._property_fields = None
            return
        
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class StartedRequest(Request):
    """
    Mark a task status as in_progress. Optionally allows to set the task's execution progress.

    :param force: If not true, call fails if the task status is not 'not_started'
    :type force: bool
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "started"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'force': {
                'default': False,
                'description': "If not true, call fails if the task status is not 'not_started'",
                'type': ['boolean', 'null'],
            },
            'status_message': {
                'description': 'Extra information regarding status change',
                'type': 'string',
            },
            'status_reason': {
                'description': 'Reason for status change',
                'type': 'string',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }
    def __init__(
            self, task, force=False, status_reason=None, status_message=None, **kwargs):
        super(StartedRequest, self).__init__(**kwargs)
        self.force = force
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property('force')
    def force(self):
        return self._property_force

    @force.setter
    def force(self, value):
        if value is None:
            self._property_force = None
            return
        
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

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

    @schema_property('status_reason')
    def status_reason(self):
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value):
        if value is None:
            self._property_status_reason = None
            return
        
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property('status_message')
    def status_message(self):
        return self._property_status_message

    @status_message.setter
    def status_message(self, value):
        if value is None:
            self._property_status_message = None
            return
        
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class StartedResponse(Response):
    """
    Response of tasks.started endpoint.

    :param started: Number of tasks started (0 or 1)
    :type started: int
    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """
    _service = "tasks"
    _action = "started"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'fields': {
                'additionalProperties': True,
                'description': 'Updated fields names and values',
                'type': ['object', 'null'],
            },
            'started': {
                'description': 'Number of tasks started (0 or 1)',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
            'updated': {
                'description': 'Number of tasks updated (0 or 1)',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, started=None, updated=None, fields=None, **kwargs):
        super(StartedResponse, self).__init__(**kwargs)
        self.started = started
        self.updated = updated
        self.fields = fields

    @schema_property('started')
    def started(self):
        return self._property_started

    @started.setter
    def started(self, value):
        if value is None:
            self._property_started = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "started", six.integer_types)
        self._property_started = value

    @schema_property('updated')
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

    @schema_property('fields')
    def fields(self):
        return self._property_fields

    @fields.setter
    def fields(self, value):
        if value is None:
            self._property_fields = None
            return
        
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class StopRequest(Request):
    """
    Request to stop a running task

    :param force: If not true, call fails if the task status is not 'in_progress'
    :type force: bool
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "stop"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'force': {
                'default': False,
                'description': "If not true, call fails if the task status is not 'in_progress'",
                'type': ['boolean', 'null'],
            },
            'status_message': {
                'description': 'Extra information regarding status change',
                'type': 'string',
            },
            'status_reason': {
                'description': 'Reason for status change',
                'type': 'string',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }
    def __init__(
            self, task, force=False, status_reason=None, status_message=None, **kwargs):
        super(StopRequest, self).__init__(**kwargs)
        self.force = force
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property('force')
    def force(self):
        return self._property_force

    @force.setter
    def force(self, value):
        if value is None:
            self._property_force = None
            return
        
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

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

    @schema_property('status_reason')
    def status_reason(self):
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value):
        if value is None:
            self._property_status_reason = None
            return
        
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property('status_message')
    def status_message(self):
        return self._property_status_message

    @status_message.setter
    def status_message(self, value):
        if value is None:
            self._property_status_message = None
            return
        
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class StopResponse(Response):
    """
    Response of tasks.stop endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """
    _service = "tasks"
    _action = "stop"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'fields': {
                'additionalProperties': True,
                'description': 'Updated fields names and values',
                'type': ['object', 'null'],
            },
            'updated': {
                'description': 'Number of tasks updated (0 or 1)',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, updated=None, fields=None, **kwargs):
        super(StopResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property('updated')
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

    @schema_property('fields')
    def fields(self):
        return self._property_fields

    @fields.setter
    def fields(self, value):
        if value is None:
            self._property_fields = None
            return
        
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class StoppedRequest(Request):
    """
    Signal a task has stopped

    :param force: If not true, call fails if the task status is not 'stopped'
    :type force: bool
    :param task: Task ID
    :type task: str
    :param status_reason: Reason for status change
    :type status_reason: str
    :param status_message: Extra information regarding status change
    :type status_message: str
    """

    _service = "tasks"
    _action = "stopped"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'force': {
                'default': False,
                'description': "If not true, call fails if the task status is not 'stopped'",
                'type': ['boolean', 'null'],
            },
            'status_message': {
                'description': 'Extra information regarding status change',
                'type': 'string',
            },
            'status_reason': {
                'description': 'Reason for status change',
                'type': 'string',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }
    def __init__(
            self, task, force=False, status_reason=None, status_message=None, **kwargs):
        super(StoppedRequest, self).__init__(**kwargs)
        self.force = force
        self.task = task
        self.status_reason = status_reason
        self.status_message = status_message

    @schema_property('force')
    def force(self):
        return self._property_force

    @force.setter
    def force(self, value):
        if value is None:
            self._property_force = None
            return
        
        self.assert_isinstance(value, "force", (bool,))
        self._property_force = value

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

    @schema_property('status_reason')
    def status_reason(self):
        return self._property_status_reason

    @status_reason.setter
    def status_reason(self, value):
        if value is None:
            self._property_status_reason = None
            return
        
        self.assert_isinstance(value, "status_reason", six.string_types)
        self._property_status_reason = value

    @schema_property('status_message')
    def status_message(self):
        return self._property_status_message

    @status_message.setter
    def status_message(self, value):
        if value is None:
            self._property_status_message = None
            return
        
        self.assert_isinstance(value, "status_message", six.string_types)
        self._property_status_message = value


class StoppedResponse(Response):
    """
    Response of tasks.stopped endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """
    _service = "tasks"
    _action = "stopped"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'fields': {
                'additionalProperties': True,
                'description': 'Updated fields names and values',
                'type': ['object', 'null'],
            },
            'updated': {
                'description': 'Number of tasks updated (0 or 1)',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, updated=None, fields=None, **kwargs):
        super(StoppedResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property('updated')
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

    @schema_property('fields')
    def fields(self):
        return self._property_fields

    @fields.setter
    def fields(self, value):
        if value is None:
            self._property_fields = None
            return
        
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class UpdateRequest(Request):
    """
    Update task's runtime parameters

    :param task: ID of the task
    :type task: str
    :param name: Task name Unique within the company.
    :type name: str
    :param tags: Tags list
    :type tags: Sequence[str]
    :param comment: Free text comment
    :type comment: str
    :param project: Project ID of the project to which this task is assigned
    :type project: str
    :param output__error: Free text error
    :type output__error: str
    :param created: Task creation time (UTC)
    :type created: datetime.datetime
    """

    _service = "tasks"
    _action = "update"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'comment': {'description': 'Free text comment ', 'type': 'string'},
            'created': {
                'description': 'Task creation time (UTC) ',
                'format': 'date-time',
                'type': 'string',
            },
            'name': {
                'description': 'Task name Unique within the company.',
                'type': 'string',
            },
            'output__error': {'description': 'Free text error', 'type': 'string'},
            'project': {
                'description': 'Project ID of the project to which this task is assigned',
                'type': 'string',
            },
            'tags': {'description': 'Tags list', 'items': {'type': 'string'}, 'type': 'array'},
            'task': {'description': 'ID of the task', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }
    def __init__(
            self, task, name=None, tags=None, comment=None, project=None, output__error=None, created=None, **kwargs):
        super(UpdateRequest, self).__init__(**kwargs)
        self.task = task
        self.name = name
        self.tags = tags
        self.comment = comment
        self.project = project
        self.output__error = output__error
        self.created = created

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

    @schema_property('comment')
    def comment(self):
        return self._property_comment

    @comment.setter
    def comment(self, value):
        if value is None:
            self._property_comment = None
            return
        
        self.assert_isinstance(value, "comment", six.string_types)
        self._property_comment = value

    @schema_property('project')
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return
        
        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property('output__error')
    def output__error(self):
        return self._property_output__error

    @output__error.setter
    def output__error(self, value):
        if value is None:
            self._property_output__error = None
            return
        
        self.assert_isinstance(value, "output__error", six.string_types)
        self._property_output__error = value

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


class UpdateResponse(Response):
    """
    Response of tasks.update endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    :param fields: Updated fields names and values
    :type fields: dict
    """
    _service = "tasks"
    _action = "update"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'fields': {
                'additionalProperties': True,
                'description': 'Updated fields names and values',
                'type': ['object', 'null'],
            },
            'updated': {
                'description': 'Number of tasks updated (0 or 1)',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, updated=None, fields=None, **kwargs):
        super(UpdateResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property('updated')
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

    @schema_property('fields')
    def fields(self):
        return self._property_fields

    @fields.setter
    def fields(self, value):
        if value is None:
            self._property_fields = None
            return
        
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class UpdateBatchRequest(BatchRequest):
    """
    Updates a batch of tasks.
            Headers
            Content type should be 'application/json-lines'.

    """

    _service = "tasks"
    _action = "update_batch"
    _version = "1.5"
    _batched_request_cls = UpdateRequest


class UpdateBatchResponse(Response):
    """
    Response of tasks.update_batch endpoint.

    :param updated: Number of tasks updated (0 or 1)
    :type updated: int
    """
    _service = "tasks"
    _action = "update_batch"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'updated': {
                'description': 'Number of tasks updated (0 or 1)',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, updated=None, **kwargs):
        super(UpdateBatchResponse, self).__init__(**kwargs)
        self.updated = updated

    @schema_property('updated')
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


class ValidateRequest(Request):
    """
    Validate task properties (before create)

    :param name: Task name. Unique within the company.
    :type name: str
    :param tags: Tags list
    :type tags: Sequence[str]
    :param type: Type of task
    :type type: TaskTypeEnum
    :param comment: Free text comment
    :type comment: str
    :param parent: Parent task id Must be a completed task.
    :type parent: str
    :param project: Project ID of the project to which this task is assigned Must
        exist[ab]
    :type project: str
    :param input: Task input params.  (input view must be provided).
    :type input: Input
    :param output_dest: Output storage id Must be a reference to an existing
        storage.
    :type output_dest: str
    :param execution: Task execution params
    :type execution: Execution
    :param script: Script info
    :type script: Script
    """

    _service = "tasks"
    _action = "validate"
    _version = "1.9"
    _schema = {
        'definitions': {
            'augmentation': {
                'properties': {
                    'crop_around_rois': {
                        'description': 'Crop image data around all frame ROIs',
                        'type': ['boolean', 'null'],
                    },
                    'sets': {
                        'description': 'List of augmentation sets',
                        'items': {'$ref': '#/definitions/augmentation_set'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'augmentation_set': {
                'properties': {
                    'arguments': {
                        'additionalProperties': {
                            'additionalProperties': True,
                            'type': 'object',
                        },
                        'description': 'Arguments dictionary per custom augmentation type.',
                        'type': ['object', 'null'],
                    },
                    'cls': {
                        'description': 'Augmentation class',
                        'type': ['string', 'null'],
                    },
                    'strength': {
                        'description': 'Augmentation strength. Range [0,).',
                        'minimum': 0,
                        'type': ['number', 'null'],
                    },
                    'types': {
                        'description': 'Augmentation type',
                        'items': {'type': 'string'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'execution': {
                'properties': {
                    'dataviews': {
                        'description': 'Additional dataviews for the task',
                        'items': {'additionalProperties': True, 'type': 'object'},
                        'type': ['array', 'null'],
                    },
                    'framework': {
                        'description': 'Framework related to the task. Case insensitive. Mandatory for Training tasks. ',
                        'type': ['string', 'null'],
                    },
                    'model': {
                        'description': 'Execution input model ID Not applicable for Register (Import) tasks',
                        'type': ['string', 'null'],
                    },
                    'model_desc': {
                        'additionalProperties': True,
                        'description': 'Json object representing the Model descriptors',
                        'type': ['object', 'null'],
                    },
                    'model_labels': {
                        'additionalProperties': {'type': 'integer'},
                        'description': "Json object representing the ids of the labels in the model.\n                The keys are the layers' names and the values are the IDs.\n                Not applicable for Register (Import) tasks.\n                Mandatory for Training tasks[z]",
                        'type': ['object', 'null'],
                    },
                    'parameters': {
                        'additionalProperties': True,
                        'description': 'Json object containing the Task parameters',
                        'type': ['object', 'null'],
                    },
                    'queue': {
                        'description': 'Queue ID where task was queued.',
                        'type': ['string', 'null'],
                    },
                    'test_split': {
                        'description': 'Percentage of frames to use for testing only',
                        'type': ['integer', 'null'],
                    },
                },
                'type': 'object',
            },
            'filter_by_roi_enum': {
                'default': 'label_rules',
                'enum': ['disabled', 'no_rois', 'label_rules'],
                'type': 'string',
            },
            'filter_label_rule': {
                'properties': {
                    'conf_range': {
                        'description': 'Range of ROI confidence level in the frame (min, max). -1 for not applicable\n            Both min and max can be either -1 or positive.\n            2nd number (max) must be either -1 or larger than or equal to the 1st number (min)',
                        'items': {'type': 'number'},
                        'maxItems': 2,
                        'minItems': 1,
                        'type': 'array',
                    },
                    'count_range': {
                        'description': 'Range of times ROI appears in the frame (min, max). -1 for not applicable.\n            Both integers must be larger than or equal to -1.\n            2nd integer (max) must be either -1 or larger than or equal to the 1st integer (min)',
                        'items': {'type': 'integer'},
                        'maxItems': 2,
                        'minItems': 1,
                        'type': 'array',
                    },
                    'label': {
                        'description': "Lucene format query (see lucene query syntax).\nDefault search field is label.keyword and default operator is AND, so searching for:\n\n'Bus Stop' Blue\n\nis equivalent to:\n\nLabel.keyword:'Bus Stop' AND label.keyword:'Blue'",
                        'type': 'string',
                    },
                },
                'required': ['label'],
                'type': 'object',
            },
            'filter_rule': {
                'properties': {
                    'dataset': {
                        'description': "Dataset ID. Must be a dataset which is in the task's view. If set to '*' all datasets in View are used.",
                        'type': 'string',
                    },
                    'filter_by_roi': {
                        '$ref': '#/definitions/filter_by_roi_enum',
                        'description': 'Type of filter',
                    },
                    'frame_query': {
                        'description': 'Frame filter, in Lucene query syntax',
                        'type': 'string',
                    },
                    'label_rules': {
                        'description': "List of FilterLabelRule ('AND' connection)\n\ndisabled - No filtering by ROIs. Select all frames, even if they don't have ROIs (all frames)\n\nno_rois - Select only frames without ROIs (empty frames)\n\nlabel_rules - Select frames according to label rules",
                        'items': {'$ref': '#/definitions/filter_label_rule'},
                        'type': ['array', 'null'],
                    },
                    'sources_query': {
                        'description': 'Sources filter, in Lucene query syntax. Filters sources in each frame.',
                        'type': 'string',
                    },
                    'version': {
                        'description': "Dataset version to apply rule to. Must belong to the dataset and be in the task's view. If set to '*' all version of the datasets in View are used.",
                        'type': 'string',
                    },
                    'weight': {
                        'description': 'Rule weight. Default is 1',
                        'type': 'number',
                    },
                },
                'required': ['filter_by_roi'],
                'type': 'object',
            },
            'filtering': {
                'properties': {
                    'filtering_rules': {
                        'description': "List of FilterRule ('OR' connection)",
                        'items': {'$ref': '#/definitions/filter_rule'},
                        'type': ['array', 'null'],
                    },
                    'output_rois': {
                        'description': "'all_in_frame' - all rois for a frame are returned\n\n'only_filtered' - only rois which led this frame to be selected\n\n'frame_per_roi' - single roi per frame. Frame can be returned multiple times with a different roi each time.\n\nNote: this should be used for Training tasks only\n\nNote: frame_per_roi implies that only filtered rois will be returned\n                ",
                        'oneOf': [
                            {'$ref': '#/definitions/output_rois_enum'},
                            {'type': 'null'},
                        ],
                    },
                },
                'type': 'object',
            },
            'input': {
                'properties': {
                    'augmentation': {
                        'description': 'Augmentation parameters. Only for training and testing tasks.',
                        'oneOf': [
                            {'$ref': '#/definitions/augmentation'},
                            {'type': 'null'},
                        ],
                    },
                    'dataviews': {
                        'additionalProperties': {'type': 'string'},
                        'description': 'Key to DataView ID Mapping',
                        'type': ['object', 'null'],
                    },
                    'frames_filter': {
                        'description': 'Filtering params',
                        'oneOf': [
                            {'$ref': '#/definitions/filtering'},
                            {'type': 'null'},
                        ],
                    },
                    'iteration': {
                        'description': 'Iteration parameters. Not applicable for register (import) tasks.',
                        'oneOf': [
                            {'$ref': '#/definitions/iteration'},
                            {'type': 'null'},
                        ],
                    },
                    'mapping': {
                        'description': 'Mapping params (see common definitions section)',
                        'oneOf': [
                            {'$ref': '#/definitions/mapping'},
                            {'type': 'null'},
                        ],
                    },
                    'view': {
                        'description': 'View params',
                        'oneOf': [{'$ref': '#/definitions/view'}, {'type': 'null'}],
                    },
                },
                'type': 'object',
            },
            'iteration': {
                'description': 'Sequential Iteration API configuration',
                'properties': {
                    'infinite': {
                        'description': 'Infinite iteration',
                        'type': ['boolean', 'null'],
                    },
                    'jump': {
                        'description': 'Jump entry',
                        'oneOf': [{'$ref': '#/definitions/jump'}, {'type': 'null'}],
                    },
                    'limit': {
                        'description': 'Maximum frames per task. If not passed, frames will end when no more matching frames are found, unless infinite is True.',
                        'type': ['integer', 'null'],
                    },
                    'min_sequence': {
                        'description': 'Length (in ms) of video clips to return. This is used in random order, and in sequential order only if jumping is provided and only for video frames',
                        'type': ['integer', 'null'],
                    },
                    'order': {
                        'description': "\n                Input frames order. Values: 'sequential', 'random'\n                In Sequential mode frames will be returned according to the order in which the frames were added to the dataset.",
                        'type': ['string', 'null'],
                    },
                    'random_seed': {
                        'description': 'Random seed used during iteration',
                        'type': 'integer',
                    },
                },
                'required': ['random_seed'],
                'type': 'object',
            },
            'jump': {
                'properties': {
                    'time': {
                        'description': 'Max time in milliseconds between frames',
                        'type': ['integer', 'null'],
                    },
                },
                'type': 'object',
            },
            'label_source': {
                'properties': {
                    'dataset': {
                        'description': "Source dataset id. '*' for all datasets in view",
                        'type': ['string', 'null'],
                    },
                    'labels': {
                        'description': "List of source labels (AND connection). '*' indicates any label. Labels must exist in at least one of the dataset versions in the task's view",
                        'items': {'type': 'string'},
                        'type': ['array', 'null'],
                    },
                    'version': {
                        'description': "Source dataset version id. Default is '*' (for all versions in dataset in the view) Version must belong to the selected dataset, and must be in the task's view[i]",
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'mapping': {
                'properties': {
                    'rules': {
                        'description': 'Rules list',
                        'items': {'$ref': '#/definitions/mapping_rule'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'mapping_rule': {
                'properties': {
                    'source': {
                        'description': 'Source label info',
                        'oneOf': [
                            {'$ref': '#/definitions/label_source'},
                            {'type': 'null'},
                        ],
                    },
                    'target': {
                        'description': 'Target label name',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'output_rois_enum': {
                'enum': ['all_in_frame', 'only_filtered', 'frame_per_roi'],
                'type': 'string',
            },
            'script': {
                'properties': {
                    'binary': {
                        'default': 'python',
                        'description': 'Binary to use when running the script',
                        'type': ['string', 'null'],
                    },
                    'branch': {
                        'description': 'Repository branch id If not provided and tag not provided, default repository branch is used.',
                        'type': ['string', 'null'],
                    },
                    'entry_point': {
                        'description': 'Path to execute within the repository',
                        'type': ['string', 'null'],
                    },
                    'repository': {
                        'description': 'Name of the repository where the script is located',
                        'type': ['string', 'null'],
                    },
                    'requirements': {
                        'description': 'A JSON object containing requirements strings by key',
                        'type': ['object', 'null'],
                    },
                    'tag': {
                        'description': 'Repository tag',
                        'type': ['string', 'null'],
                    },
                    'version_num': {
                        'description': 'Version (changeset) number. Optional (default is head version) Unused if tag is provided.',
                        'type': ['string', 'null'],
                    },
                    'working_dir': {
                        'description': 'Path to the folder from which to run the script Default - root folder of repository[f]',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
            'task_type_enum': {
                'enum': [
                    'dataset_import',
                    'annotation',
                    'annotation_manual',
                    'training',
                    'testing',
                ],
                'type': 'string',
            },
            'view': {
                'properties': {
                    'entries': {
                        'description': 'List of view entries. All tasks must have at least one view.',
                        'items': {'$ref': '#/definitions/view_entry'},
                        'type': ['array', 'null'],
                    },
                },
                'type': 'object',
            },
            'view_entry': {
                'properties': {
                    'dataset': {
                        'description': 'Existing Dataset id',
                        'type': ['string', 'null'],
                    },
                    'merge_with': {
                        'description': 'Version ID to merge with',
                        'type': ['string', 'null'],
                    },
                    'version': {
                        'description': 'Version id of a version belonging to the dataset',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
        },
        'properties': {
            'comment': {'description': 'Free text comment ', 'type': 'string'},
            'execution': {
                '$ref': '#/definitions/execution',
                'description': 'Task execution params',
            },
            'input': {
                '$ref': '#/definitions/input',
                'description': 'Task input params.  (input view must be provided).',
            },
            'name': {
                'description': 'Task name. Unique within the company.',
                'type': 'string',
            },
            'output_dest': {
                'description': 'Output storage id Must be a reference to an existing storage.',
                'type': 'string',
            },
            'parent': {
                'description': 'Parent task id Must be a completed task.',
                'type': 'string',
            },
            'project': {
                'description': 'Project ID of the project to which this task is assigned Must exist[ab]',
                'type': 'string',
            },
            'script': {
                '$ref': '#/definitions/script',
                'description': 'Script info',
            },
            'tags': {'description': 'Tags list', 'items': {'type': 'string'}, 'type': 'array'},
            'type': {
                '$ref': '#/definitions/task_type_enum',
                'description': 'Type of task',
            },
        },
        'required': ['name', 'type'],
        'type': 'object',
    }
    def __init__(
            self, name, type, tags=None, comment=None, parent=None, project=None, input=None, output_dest=None, execution=None, script=None, **kwargs):
        super(ValidateRequest, self).__init__(**kwargs)
        self.name = name
        self.tags = tags
        self.type = type
        self.comment = comment
        self.parent = parent
        self.project = project
        self.input = input
        self.output_dest = output_dest
        self.execution = execution
        self.script = script

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

    @schema_property('type')
    def type(self):
        return self._property_type

    @type.setter
    def type(self, value):
        if value is None:
            self._property_type = None
            return
        if isinstance(value, six.string_types):
            try:
                value = TaskTypeEnum(value)
            except ValueError:
                pass
        else:
            self.assert_isinstance(value, "type", enum.Enum)
        self._property_type = value

    @schema_property('comment')
    def comment(self):
        return self._property_comment

    @comment.setter
    def comment(self, value):
        if value is None:
            self._property_comment = None
            return
        
        self.assert_isinstance(value, "comment", six.string_types)
        self._property_comment = value

    @schema_property('parent')
    def parent(self):
        return self._property_parent

    @parent.setter
    def parent(self, value):
        if value is None:
            self._property_parent = None
            return
        
        self.assert_isinstance(value, "parent", six.string_types)
        self._property_parent = value

    @schema_property('project')
    def project(self):
        return self._property_project

    @project.setter
    def project(self, value):
        if value is None:
            self._property_project = None
            return
        
        self.assert_isinstance(value, "project", six.string_types)
        self._property_project = value

    @schema_property('input')
    def input(self):
        return self._property_input

    @input.setter
    def input(self, value):
        if value is None:
            self._property_input = None
            return
        if isinstance(value, dict):
            value = Input.from_dict(value)
        else:
            self.assert_isinstance(value, "input", Input)
        self._property_input = value

    @schema_property('output_dest')
    def output_dest(self):
        return self._property_output_dest

    @output_dest.setter
    def output_dest(self, value):
        if value is None:
            self._property_output_dest = None
            return
        
        self.assert_isinstance(value, "output_dest", six.string_types)
        self._property_output_dest = value

    @schema_property('execution')
    def execution(self):
        return self._property_execution

    @execution.setter
    def execution(self, value):
        if value is None:
            self._property_execution = None
            return
        if isinstance(value, dict):
            value = Execution.from_dict(value)
        else:
            self.assert_isinstance(value, "execution", Execution)
        self._property_execution = value

    @schema_property('script')
    def script(self):
        return self._property_script

    @script.setter
    def script(self, value):
        if value is None:
            self._property_script = None
            return
        if isinstance(value, dict):
            value = Script.from_dict(value)
        else:
            self.assert_isinstance(value, "script", Script)
        self._property_script = value


class ValidateResponse(Response):
    """
    Response of tasks.validate endpoint.

    """
    _service = "tasks"
    _action = "validate"
    _version = "1.9"

    _schema = {'additionalProperties': False, 'definitions': {}, 'type': 'object'}


response_mapping = {
    GetByIdRequest: GetByIdResponse,
    GetAllRequest: GetAllResponse,
    CreateRequest: CreateResponse,
    ValidateRequest: ValidateResponse,
    UpdateRequest: UpdateResponse,
    UpdateBatchRequest: UpdateBatchResponse,
    EditRequest: EditResponse,
    ResetRequest: ResetResponse,
    DeleteRequest: DeleteResponse,
    StartedRequest: StartedResponse,
    StopRequest: StopResponse,
    StoppedRequest: StoppedResponse,
    FailedRequest: FailedResponse,
    CloseRequest: CloseResponse,
    PublishRequest: PublishResponse,
    EnqueueRequest: EnqueueResponse,
    DequeueRequest: DequeueResponse,
    SetRequirementsRequest: SetRequirementsResponse,
}
