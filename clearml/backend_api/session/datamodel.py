import keyword

import enum
import json
from datetime import datetime

import jsonschema
from enum import Enum

import six


def format_date(obj):
    if isinstance(obj, datetime):
        return str(obj)


class SchemaProperty(property):
    def __init__(self, name=None, *args, **kwargs):
        super(SchemaProperty, self).__init__(*args, **kwargs)
        self.name = name

    def setter(self, fset):
        return type(self)(self.name, self.fget, fset, self.fdel, self.__doc__)


def schema_property(name):
    def init(*args, **kwargs):
        return SchemaProperty(name, *args, **kwargs)
    return init


# Support both jsonschema >= 3.0.0 and <= 2.6.0
_CustomValidator = None
try:
    from jsonschema import TypeChecker, Draft7Validator  # noqa: F401

    def _is_array(checker, instance):
        return isinstance(instance, (list, tuple))

    _CustomValidator = jsonschema.validators.extend(
        Draft7Validator,
        type_checker=Draft7Validator.TYPE_CHECKER.redefine("array", _is_array)
    )
except ImportError:
    pass


class DataModel(object):
    """ Data Model"""
    _schema = None
    _data_props_list = None

    @classmethod
    def _get_data_props(cls):
        props = cls._data_props_list
        if props is None:
            props = {}
            for c in cls.__mro__:
                props.update({k: getattr(v, 'name', k) for k, v in vars(c).items()
                              if isinstance(v, property)})
            cls._data_props_list = props
        return props.copy()

    @classmethod
    def _to_base_type(cls, value):
        if isinstance(value, dict):
            # Note: this should come before DataModel to handle data models that are simply a dict
            # (and thus are not expected to have additional named properties)
            return {k: cls._to_base_type(v) for k, v in value.items()}
        elif isinstance(value, DataModel):
            return value.to_dict()
        elif isinstance(value, enum.Enum):
            return value.value
        elif isinstance(value, list):
            return [cls._to_base_type(model) for model in value]
        return value

    def to_dict(self, only=None, except_=None):
        prop_values = {v: getattr(self, k) for k, v in self._get_data_props().items()}
        return {
            k: self._to_base_type(v)
            for k, v in prop_values.items()
            if v is not None and (not only or k in only) and (not except_ or k not in except_)
        }

    def validate(self, schema=None):
        if _CustomValidator is None:
            jsonschema.validate(
                self.to_dict(),
                schema or self._schema,
                types=dict(array=(list, tuple), integer=six.integer_types),
            )
        else:
            jsonschema.validate(
                self.to_dict(),
                schema or self._schema,
                cls=_CustomValidator,
            )

    def __repr__(self):
        return '<{}.{}: {}>'.format(
            self.__module__.split('.')[-1],
            type(self).__name__,
            json.dumps(
                self.to_dict(),
                indent=4,
                default=format_date,
            )
        )

    @staticmethod
    def assert_isinstance(value, field_name, expected, is_array=False):
        if not is_array:
            if not isinstance(value, expected):
                raise TypeError("Expected %s of type %s, got %s" % (field_name, expected, type(value).__name__))
            return

        if not all(isinstance(x, expected) for x in value):
            raise TypeError(
                "Expected %s of type list[%s], got %s" % (
                    field_name,
                    expected,
                    ", ".join(set(type(x).__name__ for x in value)),
                )
            )

    @staticmethod
    def normalize_key(prop_key):
        if keyword.iskeyword(prop_key):
            prop_key += '_'
        return prop_key.replace('.', '__')

    @classmethod
    def from_dict(cls, dct, strict=False):
        """
        Create an instance from a dictionary while ignoring unnecessary keys
        """
        allowed_keys = cls._get_data_props().values()
        invalid_keys = set(dct).difference(allowed_keys)
        if strict and invalid_keys:
            raise ValueError("Invalid keys %s" % tuple(invalid_keys))
        return cls(**{cls.normalize_key(key): value for key, value in dct.items() if key not in invalid_keys})


class UnusedKwargsWarning(UserWarning):
    pass


class NonStrictDataModelMixin(object):
    """
    NonStrictDataModelMixin

    :summary: supplies an __init__ method that warns about unused keywords
    """

    def __init__(self, **kwargs):
        # unexpected = [key for key in kwargs if not key.startswith('_')]
        # if unexpected:
        #     message = '{}: unused keyword argument(s) {}' \
        #         .format(type(self).__name__, unexpected)
        #     warnings.warn(message, UnusedKwargsWarning)

        # ignore extra data warnings
        pass


class NonStrictDataModel(DataModel, NonStrictDataModelMixin):
    pass


class StringEnum(Enum):

    def __str__(self):
        return self.value

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_
