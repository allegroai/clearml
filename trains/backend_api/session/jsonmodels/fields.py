import datetime
import re
from weakref import WeakKeyDictionary

import six
from dateutil.parser import parse

from .errors import ValidationError
from .collections import ModelCollection


# unique marker for "no default value specified". None is not good enough since
# it is a completely valid default value.
NotSet = object()


class BaseField(object):

    """Base class for all fields."""

    types = None

    def __init__(
            self,
            required=False,
            nullable=False,
            help_text=None,
            validators=None,
            default=NotSet,
            name=None):
        self.memory = WeakKeyDictionary()
        self.required = required
        self.help_text = help_text
        self.nullable = nullable
        self._assign_validators(validators)
        self.name = name
        self._validate_name()
        if default is not NotSet:
            self.validate(default)
        self._default = default

    @property
    def has_default(self):
        return self._default is not NotSet

    def _assign_validators(self, validators):
        if validators and not isinstance(validators, list):
            validators = [validators]
        self.validators = validators or []

    def __set__(self, instance, value):
        self._finish_initialization(type(instance))
        value = self.parse_value(value)
        self.validate(value)
        self.memory[instance._cache_key] = value

    def __get__(self, instance, owner=None):
        if instance is None:
            self._finish_initialization(owner)
            return self

        self._finish_initialization(type(instance))

        self._check_value(instance)
        return self.memory[instance._cache_key]

    def _finish_initialization(self, owner):
        pass

    def _check_value(self, obj):
        if obj._cache_key not in self.memory:
            self.__set__(obj, self.get_default_value())

    def validate_for_object(self, obj):
        value = self.__get__(obj)
        self.validate(value)

    def validate(self, value):
        self._check_types()
        self._validate_against_types(value)
        self._check_against_required(value)
        self._validate_with_custom_validators(value)

    def _check_against_required(self, value):
        if value is None and self.required:
            raise ValidationError('Field is required!')

    def _validate_against_types(self, value):
        if value is not None and not isinstance(value, self.types):
            raise ValidationError(
                'Value is wrong, expected type "{types}"'.format(
                    types=', '.join([t.__name__ for t in self.types])
                ),
                value,
            )

    def _check_types(self):
        if self.types is None:
            raise ValidationError(
                'Field "{type}" is not usable, try '
                'different field type.'.format(type=type(self).__name__))

    def to_struct(self, value):
        """Cast value to Python structure."""
        return value

    def parse_value(self, value):
        """Parse value from primitive to desired format.

        Each field can parse value to form it wants it to be (like string or
        int).

        """
        return value

    def _validate_with_custom_validators(self, value):
        if value is None and self.nullable:
            return

        for validator in self.validators:
            try:
                validator.validate(value)
            except AttributeError:
                validator(value)

    def get_default_value(self):
        """Get default value for field.

        Each field can specify its default.

        """
        return self._default if self.has_default else None

    def _validate_name(self):
        if self.name is None:
            return
        if not re.match('^[A-Za-z_](([\w\-]*)?\w+)?$', self.name):  # noqa: W605
            raise ValueError('Wrong name', self.name)

    def structue_name(self, default):
        return self.name if self.name is not None else default


class StringField(BaseField):

    """String field."""

    types = six.string_types


class IntField(BaseField):

    """Integer field."""

    types = (int,)

    def parse_value(self, value):
        """Cast value to `int`, e.g. from string or long"""
        parsed = super(IntField, self).parse_value(value)
        if parsed is None:
            return parsed
        return int(parsed)


class FloatField(BaseField):

    """Float field."""

    types = (float, int)


class BoolField(BaseField):

    """Bool field."""

    types = (bool,)

    def parse_value(self, value):
        """Cast value to `bool`."""
        parsed = super(BoolField, self).parse_value(value)
        return bool(parsed) if parsed is not None else None


class ListField(BaseField):

    """List field."""

    types = (list,)

    def __init__(self, items_types=None, *args, **kwargs):
        """Init.

        `ListField` is **always not required**. If you want to control number
        of items use validators.

        """
        self._assign_types(items_types)
        super(ListField, self).__init__(*args, **kwargs)
        self.required = False

    def get_default_value(self):
        default = super(ListField, self).get_default_value()
        if default is None:
            return ModelCollection(self)
        return default

    def _assign_types(self, items_types):
        if items_types:
            try:
                self.items_types = tuple(items_types)
            except TypeError:
                self.items_types = items_types,
        else:
            self.items_types = tuple()

        types = []
        for type_ in self.items_types:
            if isinstance(type_, six.string_types):
                types.append(_LazyType(type_))
            else:
                types.append(type_)
        self.items_types = tuple(types)

    def validate(self, value):
        super(ListField, self).validate(value)

        if len(self.items_types) == 0:
            return

        for item in value:
            self.validate_single_value(item)

    def validate_single_value(self, item):
        if len(self.items_types) == 0:
            return

        if not isinstance(item, self.items_types):
            raise ValidationError(
                'All items must be instances '
                'of "{types}", and not "{type}".'.format(
                    types=', '.join([t.__name__ for t in self.items_types]),
                    type=type(item).__name__,
                ))

    def parse_value(self, values):
        """Cast value to proper collection."""
        result = self.get_default_value()

        if not values:
            return result

        if not isinstance(values, list):
            return values

        return [self._cast_value(value) for value in values]

    def _cast_value(self, value):
        if isinstance(value, self.items_types):
            return value
        else:
            if len(self.items_types) != 1:
                tpl = 'Cannot decide which type to choose from "{types}".'
                raise ValidationError(
                    tpl.format(
                        types=', '.join([t.__name__ for t in self.items_types])
                    )
                )
            return self.items_types[0](**value)

    def _finish_initialization(self, owner):
        super(ListField, self)._finish_initialization(owner)

        types = []
        for type in self.items_types:
            if isinstance(type, _LazyType):
                types.append(type.evaluate(owner))
            else:
                types.append(type)
        self.items_types = tuple(types)

    def _elem_to_struct(self, value):
        try:
            return value.to_struct()
        except AttributeError:
            return value

    def to_struct(self, values):
        return [self._elem_to_struct(v) for v in values]


class EmbeddedField(BaseField):

    """Field for embedded models."""

    def __init__(self, model_types, *args, **kwargs):
        self._assign_model_types(model_types)
        super(EmbeddedField, self).__init__(*args, **kwargs)

    def _assign_model_types(self, model_types):
        if not isinstance(model_types, (list, tuple)):
            model_types = (model_types,)

        types = []
        for type_ in model_types:
            if isinstance(type_, six.string_types):
                types.append(_LazyType(type_))
            else:
                types.append(type_)
        self.types = tuple(types)

    def _finish_initialization(self, owner):
        super(EmbeddedField, self)._finish_initialization(owner)

        types = []
        for type in self.types:
            if isinstance(type, _LazyType):
                types.append(type.evaluate(owner))
            else:
                types.append(type)
        self.types = tuple(types)

    def validate(self, value):
        super(EmbeddedField, self).validate(value)
        try:
            value.validate()
        except AttributeError:
            pass

    def parse_value(self, value):
        """Parse value to proper model type."""
        if not isinstance(value, dict):
            return value

        embed_type = self._get_embed_type()
        return embed_type(**value)

    def _get_embed_type(self):
        if len(self.types) != 1:
            raise ValidationError(
                'Cannot decide which type to choose from "{types}".'.format(
                    types=', '.join([t.__name__ for t in self.types])
                )
            )
        return self.types[0]

    def to_struct(self, value):
        return value.to_struct()


class _LazyType(object):

    def __init__(self, path):
        self.path = path

    def evaluate(self, base_cls):
        module, type_name = _evaluate_path(self.path, base_cls)
        return _import(module, type_name)


def _evaluate_path(relative_path, base_cls):
    base_module = base_cls.__module__

    modules = _get_modules(relative_path, base_module)

    type_name = modules.pop()
    module = '.'.join(modules)
    if not module:
        module = base_module
    return module, type_name


def _get_modules(relative_path, base_module):
    canonical_path = relative_path.lstrip('.')
    canonical_modules = canonical_path.split('.')

    if not relative_path.startswith('.'):
        return canonical_modules

    parents_amount = len(relative_path) - len(canonical_path)
    parent_modules = base_module.split('.')
    parents_amount = max(0, parents_amount - 1)
    if parents_amount > len(parent_modules):
        raise ValueError("Can't evaluate path '{}'".format(relative_path))
    return parent_modules[:parents_amount * -1] + canonical_modules


def _import(module_name, type_name):
    module = __import__(module_name, fromlist=[type_name])
    try:
        return getattr(module, type_name)
    except AttributeError:
        raise ValueError(
            "Can't find type '{}.{}'.".format(module_name, type_name))


class TimeField(StringField):

    """Time field."""

    types = (datetime.time,)

    def __init__(self, str_format=None, *args, **kwargs):
        """Init.

        :param str str_format: Format to cast time to (if `None` - casting to
            ISO 8601 format).

        """
        self.str_format = str_format
        super(TimeField, self).__init__(*args, **kwargs)

    def to_struct(self, value):
        """Cast `time` object to string."""
        if self.str_format:
            return value.strftime(self.str_format)
        return value.isoformat()

    def parse_value(self, value):
        """Parse string into instance of `time`."""
        if value is None:
            return value
        if isinstance(value, datetime.time):
            return value
        return parse(value).timetz()


class DateField(StringField):

    """Date field."""

    types = (datetime.date,)
    default_format = '%Y-%m-%d'

    def __init__(self, str_format=None, *args, **kwargs):
        """Init.

        :param str str_format: Format to cast date to (if `None` - casting to
            %Y-%m-%d format).

        """
        self.str_format = str_format
        super(DateField, self).__init__(*args, **kwargs)

    def to_struct(self, value):
        """Cast `date` object to string."""
        if self.str_format:
            return value.strftime(self.str_format)
        return value.strftime(self.default_format)

    def parse_value(self, value):
        """Parse string into instance of `date`."""
        if value is None:
            return value
        if isinstance(value, datetime.date):
            return value
        return parse(value).date()


class DateTimeField(StringField):

    """Datetime field."""

    types = (datetime.datetime,)

    def __init__(self, str_format=None, *args, **kwargs):
        """Init.

        :param str str_format: Format to cast datetime to (if `None` - casting
            to ISO 8601 format).

        """
        self.str_format = str_format
        super(DateTimeField, self).__init__(*args, **kwargs)

    def to_struct(self, value):
        """Cast `datetime` object to string."""
        if self.str_format:
            return value.strftime(self.str_format)
        return value.isoformat()

    def parse_value(self, value):
        """Parse string into instance of `datetime`."""
        if isinstance(value, datetime.datetime):
            return value
        if value:
            return parse(value)
        else:
            return None
