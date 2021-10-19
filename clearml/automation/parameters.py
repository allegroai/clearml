import sys
from itertools import product
from random import Random as BaseRandom
from typing import Mapping, Any, Sequence, Optional, Union


class RandomSeed(object):
    """
    The base class controlling random sampling for every optimization strategy.
    """
    _random = BaseRandom(1337)
    _seed = 1337

    @staticmethod
    def set_random_seed(seed=1337):
        # type: (int) -> ()
        """
        Set global seed for all hyper-parameter strategy random number sampling.

        :param int seed: The random seed.
        """
        RandomSeed._seed = seed
        RandomSeed._random = BaseRandom(seed)

    @staticmethod
    def get_random_seed():
        # type: () -> int
        """
        Get the global seed for all hyper-parameter strategy random number sampling.

        :return: The random seed.
        """
        return RandomSeed._seed


class Parameter(RandomSeed):
    """
    The base hyper-parameter optimization object.
    """
    _class_type_serialize_name = 'type'

    def __init__(self, name):
        # type: (Optional[str]) -> ()
        """
        Create a new Parameter for hyper-parameter optimization

        :param str name: The new Parameter name. This is the parameter name that will be passed to a Task.
        """
        self.name = name

    def get_value(self):
        # type: () -> Mapping[str, Any]
        """
        Return a dict with the Parameter name and a sampled value for the Parameter.

        :return:

            For example:

            .. code-block:: py

                {'answer': 0.42}

        """
        pass

    def to_list(self):
        # type: () -> Sequence[Mapping[str, Any]]
        """
        Return a list of all the valid values of the Parameter.

        :return: List of dicts {name: value}
        """
        pass

    def to_dict(self):
        # type: () -> Mapping[str, Union[str, Parameter]]
        """
        Return a dict representation of the Parameter object. Used for serialization of the Parameter object.

        :return:  dict representation of the object (serialization).
        """
        serialize = {self._class_type_serialize_name: str(self.__class__).split('.')[-1][:-2]}
        # noinspection PyCallingNonCallable
        serialize.update(dict(((k, v.to_dict() if hasattr(v, 'to_dict') else v) for k, v in self.__dict__.items())))
        return serialize

    @classmethod
    def from_dict(cls, a_dict):
        # type: (Mapping[str, str]) -> Parameter
        """
        Construct Parameter object from a dict representation (deserialize from dict).

        :return:  The Parameter object.
        """
        a_dict = a_dict.copy()
        a_cls = a_dict.pop(cls._class_type_serialize_name, None)
        if not a_cls:
            return None
        try:
            a_cls = getattr(sys.modules[__name__], a_cls)
        except AttributeError:
            return None
        instance = a_cls.__new__(a_cls)
        instance.__dict__ = dict(
            (k, cls.from_dict(v) if isinstance(v, dict) and cls._class_type_serialize_name in v else v)
            for k, v in a_dict.items())
        return instance


class UniformParameterRange(Parameter):
    """
    Uniform randomly sampled hyper-parameter object.
    """

    def __init__(
            self,
            name,  # type: str
            min_value,  # type: float
            max_value,  # type: float
            step_size=None,  # type: Optional[float]
            include_max_value=True  # type: bool
    ):
        # type: (...) -> ()
        """
        Create a parameter to be sampled by the SearchStrategy

        :param str name: The parameter name. Match the Task hyper-parameter name.
        :param float min_value: The minimum sample to use for uniform random sampling.
        :param float max_value: The maximum sample to use for uniform random sampling.
        :param float step_size: If not ``None``, set step size (quantization) for value sampling.
        :param bool include_max_value: Range includes the ``max_value``

            The values are:

            - ``True`` - The range includes the ``max_value`` (Default)
            - ``False`` -  Does not include.

        """
        super(UniformParameterRange, self).__init__(name=name)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.step_size = float(step_size) if step_size is not None else None
        self.include_max = include_max_value

    def get_value(self):
        # type: () -> Mapping[str, Any]
        """
        Return uniformly sampled value based on object sampling definitions.

        :return: {self.name: random value [self.min_value, self.max_value)}
        """
        if not self.step_size:
            return {self.name: self._random.uniform(self.min_value, self.max_value)}
        steps = (self.max_value - self.min_value) / self.step_size
        return {self.name: self.min_value + (self._random.randrange(start=0, stop=round(steps)) * self.step_size)}

    def to_list(self):
        # type: () -> Sequence[Mapping[str, float]]
        """
        Return a list of all the valid values of the Parameter. If ``self.step_size`` is not defined, return 100 points
        between min/max values.

        :return: list of dicts {name: float}
        """
        step_size = self.step_size or (self.max_value - self.min_value) / 100.
        steps = (self.max_value - self.min_value) / self.step_size
        values = [v*step_size for v in range(0, int(steps))]
        if self.include_max and (not values or values[-1] < self.max_value):
            values.append(self.max_value)
        return [{self.name: v} for v in values]


class LogUniformParameterRange(UniformParameterRange):
    """
    Logarithmic uniform randomly sampled hyper-parameter object.
    """

    def __init__(
            self,
            name,  # type: str
            min_value,  # type: float
            max_value,  # type: float
            base=10,  # type: float
            step_size=None,  # type: Optional[float]
            include_max_value=True  # type: bool
    ):
        # type: (...) -> ()
        """
        Create a parameter to be sampled by the SearchStrategy

        :param str name: The parameter name. Match the Task hyper-parameter name.
        :param float min_value: The minimum exponent sample to use for uniform random sampling.
        :param float max_value: The maximum exponent sample to use for uniform random sampling.
        :param float base: The base used to raise the sampled exponent.
        :param float step_size: If not ``None``, set step size (quantization) for value sampling.
        :param bool include_max_value: Range includes the ``max_value``

            The values are:

            - ``True`` - The range includes the ``max_value`` (Default)
            - ``False`` -  Does not include.

        """
        super().__init__(name, min_value, max_value, step_size=step_size, include_max_value=include_max_value)
        self.base = base

    def get_value(self):
        """
        Return uniformly logarithmic sampled value based on object sampling definitions.

        :return: {self.name: random value self.base^[self.min_value, self.max_value)}
        """
        values_dict = super().get_value()
        return {self.name: self.base**v for v in values_dict.values()}

    def to_list(self):
        values_list = super().to_list()
        return [{self.name: self.base**v[self.name]} for v in values_list]


class UniformIntegerParameterRange(Parameter):
    """
    Uniform randomly sampled integer Hyper-Parameter object.
    """

    def __init__(self, name, min_value, max_value, step_size=1, include_max_value=True):
        # type: (str, int, int, int, bool) -> ()
        """
        Create a parameter to be sampled by the SearchStrategy.

        :param str name: The parameter name. Match the task hyper-parameter name.
        :param int min_value: The minimum sample to use for uniform random sampling.
        :param int max_value: The maximum sample to use for uniform random sampling.
        :param int step_size: The default step size is ``1``.
        :param bool include_max_value: Range includes the ``max_value``

            The values are:

            - ``True`` - Includes the ``max_value`` (Default)
            - ``False`` - Does not include.

        """
        super(UniformIntegerParameterRange, self).__init__(name=name)
        self.min_value = int(min_value)
        self.max_value = int(max_value)
        self.step_size = int(step_size) if step_size is not None else None
        self.include_max = include_max_value

    def get_value(self):
        # type: () -> Mapping[str, Any]
        """
        Return uniformly sampled value based on object sampling definitions.

        :return: {self.name: random value [self.min_value, self.max_value)}
        """
        return {self.name: self._random.randrange(
            start=self.min_value, step=self.step_size,
            stop=self.max_value + (0 if not self.include_max else self.step_size))}

    def to_list(self):
        # type: () -> Sequence[Mapping[str, int]]
        """
        Return a list of all the valid values of the Parameter. If ``self.step_size`` is not defined, return 100 points
        between minmax values.

        :return: list of dicts {name: int}
        """
        values = list(range(self.min_value, self.max_value, self.step_size))
        if self.include_max and (not values or values[-1] < self.max_value):
            values.append(self.max_value)
        return [{self.name: v} for v in values]


class DiscreteParameterRange(Parameter):
    """
    Discrete randomly sampled hyper-parameter object.
    """

    def __init__(self, name, values=()):
        # type: (str, Sequence[Any]) -> ()
        """
        Uniformly sample values form a list of discrete options.

        :param str name: The parameter name. Match the task hyper-parameter name.
        :param list values: The list/tuple of valid parameter values to sample from.
        """
        super(DiscreteParameterRange, self).__init__(name=name)
        self.values = values

    def get_value(self):
        # type: () -> Mapping[str, Any]
        """
        Return uniformly sampled value from the valid list of values.

        :return: {self.name: random entry from self.value}
        """
        return {self.name: self._random.choice(self.values)}

    def to_list(self):
        # type: () -> Sequence[Mapping[str, Any]]
        """
        Return a list of all the valid values of the Parameter.

        :return: list of dicts {name: value}
        """
        return [{self.name: v} for v in self.values]


class ParameterSet(Parameter):
    """
    Discrete randomly sampled Hyper-Parameter object.
    """

    def __init__(self, parameter_combinations=()):
        # type: (Sequence[Mapping[str, Union[float, int, str, Parameter]]]) -> ()
        """
        Uniformly sample values form a list of discrete options (combinations) of parameters.

        :param list parameter_combinations: The list/tuple of valid parameter combinations.

            For example, two combinations with three specific parameters per combination:

            .. code-block:: javascript

               [ {'opt1': 10, 'arg2': 20, 'arg2': 30},
                 {'opt2': 11, 'arg2': 22, 'arg2': 33}, ]

            Two complex combination each one sampled from a different range:

            .. code-block:: javascript

               [ {'opt1': UniformParameterRange('arg1',0,1) , 'arg2': 20},
                 {'opt2': UniformParameterRange('arg1',11,12), 'arg2': 22},]
        """
        super(ParameterSet, self).__init__(name=None)
        self.values = parameter_combinations

    def get_value(self):
        # type: () -> Mapping[str, Any]
        """
        Return uniformly sampled value from the valid list of values.

        :return: {self.name: random entry from self.value}
        """
        return self._get_value(self._random.choice(self.values))

    def to_list(self):
        # type: () -> Sequence[Mapping[str, Any]]
        """
        Return a list of all the valid values of the Parameter.

        :return: list of dicts {name: value}
        """
        combinations = []
        for combination in self.values:
            single_option = {}
            for k, v in combination.items():
                if isinstance(v, Parameter):
                    single_option[k] = v.to_list()
                else:
                    single_option[k] = [{k: v}, ]

            for state in product(*single_option.values()):
                combinations.append(dict(kv for d in state for kv in d.items()))

        return combinations

    @staticmethod
    def _get_value(combination):
        # type: (dict) -> dict
        value_dict = {}
        for k, v in combination.items():
            if isinstance(v, Parameter):
                value_dict.update(v.get_value())
            else:
                value_dict[k] = v

        return value_dict
