import re
import typing
from collections import OrderedDict

try:
    from collections import UserList
except ImportError:
    UserList = list

try:
    from collections import UserDict
except ImportError:
    UserDict = dict

from clearml.backend_api import Session
from clearml.backend_api.services import models


class ModelsList(UserList):
    def __init__(self, models_dict):
        # type: (typing.OrderedDict["clearml.Model"]) -> None # noqa: F821
        self._models = models_dict
        super(ModelsList, self).__init__(models_dict.values())

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._models[item]
        return super(ModelsList, self).__getitem__(item)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class TaskModels(UserDict):
    _input_models_re = re.compile(pattern=r"((Using model id: )(\w+)?)", flags=re.IGNORECASE)

    @property
    def input(self):
        # type: () -> ModelsList
        return self._input

    @property
    def output(self):
        # type: () -> ModelsList
        return self._output

    def __init__(self, task):
        # type: ("clearml.Task") -> None # noqa: F821
        self._input = self._get_input_models(task)
        self._output = self._get_output_models(task)

        super(TaskModels, self).__init__({"input": self._input, "output": self._output})

    def _get_input_models(self, task):
        # type: ("clearml.Task") -> ModelsList # noqa: F821

        if Session.check_min_api_version("2.13"):
            parsed_ids = list(task.input_models_id.values())
        else:
            # since we'll fall back to the new task.models.input if no parsed IDs are found, only
            #  extend this with the input model in case we're using 2.13 and have any parsed IDs or if we're using
            #  a lower API version.
            parsed_ids = [i[-1] for i in self._input_models_re.findall(task.comment)]
            # get the last one on the Task
            parsed_ids.extend(list(task.input_models_id.values()))

        from clearml.model import Model

        def get_model(id_):
            m = Model(model_id=id_)
            # noinspection PyBroadException
            try:
                # make sure the model is is valid
                # noinspection PyProtectedMember
                m._get_model_data()
                return m
            except Exception:
                pass

        # noinspection PyProtectedMember
        if Session.check_min_api_version("2.13") and task._get_task_property(
                "models.input", raise_on_error=False, log_on_error=False):
            input_models = OrderedDict(
                (x.name, get_model(x.model)) for x in task.data.models.input
            )
        else:
            # remove duplicates and preserve order
            input_models = OrderedDict(
                ("Input Model #{}".format(i), a_model)
                for i, a_model in enumerate(
                    filter(None, map(get_model, OrderedDict.fromkeys(parsed_ids)))
                )
            )

        return ModelsList(input_models)

    @staticmethod
    def _get_output_models(task):
        # type: ("clearml.Task") -> ModelsList # noqa: F821

        res = task.send(
            models.GetAllRequest(
                task=[task.id], order_by=["created"], only_fields=["id"]
            )
        )
        ids = [m.id for m in res.response.models or []] + list(task.output_models_id.values())
        # remove duplicates and preserve order
        ids = list(OrderedDict.fromkeys(ids))

        id_to_name = (
            {x.model: x.name for x in task.data.models.output}
            if Session.check_min_api_version("2.13")
            else {}
        )

        def resolve_name(index, model_id):
            return id_to_name.get(model_id, "Output Model #{}".format(index))

        from clearml.model import Model

        output_models = OrderedDict(
            (resolve_name(i, m_id), Model(model_id=m_id)) for i, m_id in enumerate(ids)
        )

        return ModelsList(output_models)
