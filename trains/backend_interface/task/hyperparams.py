from collections import defaultdict
from typing import Optional, Any, Sequence, Callable, Mapping, Union, Dict, Iterable, Generator

from ...backend_api import Session
from ...backend_api.services import tasks


class HyperParams(object):
    def __init__(self, task):
        self.task = task

    def get_hyper_params(
        self,
        sections=None,  # type: Optional[Sequence[str]]
        selector=None,  # type: Optional[Callable[[dict], bool]]
        projector=None,  # type: Optional[Callable[[dict], Any]]
        return_obj=False  # type: Optional[bool]
    ):
        # type: (...) -> Dict[str, Dict[Union[dict, Any]]]
        """
        Get hyper-parameters for this task.
        Returns a dictionary mapping user property name to user property details dict.
        :param sections: Return only hyper-params in the provided sections
        :param selector: A callable selecting which hyper-parameters should be returned
        :param projector: A callable to project values before they are returned
        :param return_obj: If True, returned dictionary values are API objects (tasks.ParamsItem). If ``projeictor
        """
        if not Session.check_min_api_version("2.9"):
            raise ValueError("Not supported by server")

        task_id = self.task.task_id

        res = self.task.session.send(tasks.GetHyperParamsRequest(tasks=[task_id]))
        hyperparams = defaultdict(defaultdict)
        if res.ok() and res.response.params:
            for entry in res.response.params:
                if entry.get("task") == task_id:
                    for item in entry.get("hyperparams", []):
                        # noinspection PyBroadException
                        try:
                            if (sections and item.get("section") not in sections) or (
                                selector and not selector(item)
                            ):
                                continue
                            item = item if not projector else projector(item)
                            if return_obj:
                                item = tasks.ParamsItem()
                            hyperparams[item.get("section")][item.get("name")] = (
                                item if not projector else projector(item)
                            )
                        except Exception:
                            self.task.log.exception("Failed processing hyper-parameter")
        return hyperparams

    def edit_hyper_params(
        self,
        iterables,  # type: Union[Mapping[str, Union[str, dict, None]], Iterable[dict, tasks.ParamsItem]]
        replace=None,  # type: Optional[str]
        default_section=None,  # type: Optional[str]
        force_section=None  # type: Optional[str]
    ):
        # type: (...) -> bool
        """
        Set hyper-parameters for this task.
        :param iterables: Hyper parameter iterables, each can be:
            * A dictionary of string key (name) to either a string value (value), a tasks.ParamsItem or a dict
             (hyperparam details). If ``default_section`` is not provided, each dict must contain a "section" field.
            * An iterable of tasks.ParamsItem or dicts (each representing hyperparam details).
              Each dict must contain a "name" field. If ``default_section`` is not provided, each dict must
            also contain a "section" field.
        :param replace: Optional replace strategy, values are:
            * 'all' - provided hyper-params replace all existing hyper-params in task
            * 'section' - only sections present in the provided hyper-params are replaced
            * 'none' (default) - provided hyper-params will be merged into existing task hyper-params (i.e. will be
              added or update existing hyper-params)
        :param default_section: Optional section name to be used when section is not explicitly provided.
        :param force_section: Optional section name to be used for all hyper-params.
        """
        if not Session.check_min_api_version("2.9"):
            raise ValueError("Not supported by server")

        escape_unsafe = not Session.check_min_api_version("2.11")

        if not tasks.ReplaceHyperparamsEnum.has_value(replace):
            replace = None

        def make_item(value, name=None):
            if isinstance(value, tasks.ParamsItem):
                a_item = value
            elif isinstance(value, dict):
                a_item = tasks.ParamsItem(**{k: None if v is None else str(v) for k, v in value.items()})
            elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict) and 'value' in value[1]:
                a_item = tasks.ParamsItem(
                    name=str(value[0]), **{k: None if v is None else str(v) for k, v in value[1].items()})
            elif isinstance(value, tuple):
                a_item = tasks.ParamsItem(name=str(value[0]), value=str(value[1]))
            else:
                a_item = tasks.ParamsItem(value=str(value))

            if name:
                a_item.name = str(name)
            if not a_item.name:
                raise ValueError("Missing hyper-param name for '{}'".format(value))
            section = force_section or a_item.section or default_section
            if not section:
                raise ValueError("Missing hyper-param section for '{}'".format(value))
            # force string value
            if escape_unsafe:
                a_item.section, a_item.name = self._escape_unsafe_values(section, a_item.name)
            else:
                a_item.section = section
            return a_item

        props = {}
        if isinstance(iterables, dict):
            props.update({name: make_item(name=name, value=value) for name, value in iterables.items()})
        else:
            for i in iterables:
                item = make_item(i)
                props.update({item.name: item})

        res = self.task.session.send(
            tasks.EditHyperParamsRequest(
                task=self.task.task_id,
                hyperparams=props.values(),
                replace_hyperparams=replace,
            ),
        )
        if res.ok():
            self.task.reload()
            return True

        return False

    def delete_hyper_params(self, *iterables):
        # type: (Iterable[Union[dict, Iterable[str, str], tasks.ParamKey, tasks.ParamsItem]]) -> bool
        """
        Delete hyper-parameters for this task.
        :param iterables: Hyper parameter key iterables. Each an iterable whose possible values each represent
         a hyper-parameter entry to delete, value formats are:
            * A dictionary containing a 'section' and 'name' fields
            * An iterable (e.g. tuple, list etc.) whose first two items denote 'section' and 'name'
            * An API object of type tasks.ParamKey or tasks.ParamsItem whose section and name fields are not empty
        """
        if not Session.check_min_api_version("2.9"):
            raise ValueError("Not supported by server")

        def get_key(value):
            if isinstance(value, dict):
                key = (value.get("section"), value.get("name"))
            elif isinstance(value, (tasks.ParamKey, tasks.ParamsItem)):
                key = (value.section, value.name)
            else:
                key = tuple(map(str, value))[:2]
            if not all(key):
                raise ValueError("Missing section or name in '{}'".format(value))
            return key

        keys = {get_key(value) for iterable in iterables for value in iterable}

        res = self.task.session.send(
            tasks.DeleteHyperParamsRequest(
                task=self.task.task_id,
                hyperparams=[tasks.ParamKey(section=section, name=name) for section, name in keys],
            ),
        )
        if res.ok():
            self.task.reload()
            return True

        return False

    def _escape_unsafe_values(self, *values):
        # type: (str) -> Generator[str]
        """ Escape unsafe values (name, section name) for API version 2.10 and below """
        for value in values:
            if value not in UNSAFE_NAMES_2_10:
                yield value
            else:
                self.task.log.info(
                    "Converting unsafe hyper parameter name/section '{}' to '{}'".format(value, "_" + value)
                )
                yield "_" + value


UNSAFE_NAMES_2_10 = {
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "nin",
    "mod",
    "all",
    "size",
    "exists",
    "not",
    "elemMatch",
    "type",
    "within_distance",
    "within_spherical_distance",
    "within_box",
    "within_polygon",
    "near",
    "near_sphere",
    "max_distance",
    "min_distance",
    "geo_within",
    "geo_within_box",
    "geo_within_polygon",
    "geo_within_center",
    "geo_within_sphere",
    "geo_intersects",
    "contains",
    "icontains",
    "startswith",
    "istartswith",
    "endswith",
    "iendswith",
    "exact",
    "iexact",
    "match",
}
