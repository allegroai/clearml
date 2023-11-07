import importlib
import pkgutil
import re
from typing import Any, Optional

from .session import Session
from ..utilities.check_updates import Version


class ApiServiceProxy(object):
    _main_services_module = "clearml.backend_api.services"
    _available_versions = None

    def __init__(self, module):
        self.__wrapped_name__ = module
        self.__wrapped_version__ = Session.api_version

    def __getattr__(self, attr):
        if attr in ["__wrapped_name__", "__wrapped__", "__wrapped_version__"]:
            return self.__dict__.get(attr)

        if not self.__dict__.get("__wrapped__") or self.__dict__.get("__wrapped_version__") != Session.api_version:
            if not ApiServiceProxy._available_versions:
                services = self._import_module(self._main_services_module, None)
                ApiServiceProxy._available_versions = sorted(
                    Version(name[1:].replace("_", "."))
                    for name in [
                        module_name
                        for _, module_name, _ in pkgutil.iter_modules(services.__path__)
                        if re.match(r"^v[0-9]+_[0-9]+$", module_name)
                    ]
                )

            # get the most advanced service version that supports our api
            version = [
                str(v) for v in ApiServiceProxy._available_versions
                if Session.check_min_api_version(v)
            ][-1]
            Session.api_version = version
            self.__dict__["__wrapped_version__"] = Session.api_version
            name = ".v{}.{}".format(
                version.replace(".", "_"), self.__dict__.get("__wrapped_name__")
            )
            self.__dict__["__wrapped__"] = self._import_module(name, self._main_services_module)

        return getattr(self.__dict__["__wrapped__"], attr)

    def _import_module(self, name, package):
        # type: (str, Optional[str]) -> Any
        # noinspection PyBroadException
        try:
            return importlib.import_module(name, package=package)
        except Exception:
            return None


class ExtApiServiceProxy(ApiServiceProxy):
    _extra_services_modules = []

    def _import_module(self, name, _):
        # type: (str, Optional[str]) -> Any
        for module_path in self._get_services_modules():
            try:
                return importlib.import_module(name, package=module_path)
            except ImportError:
                pass

        raise ImportError(
            "No module '{}' in all predefined services module paths".format(name)
        )

    @classmethod
    def add_services_module(cls, module_path):
        # type: (str) -> None
        """
        Add an additional service module path to look in when importing types
        """
        cls._extra_services_modules.append(module_path)

    def _get_services_modules(self):
        """
        Yield all services module paths.
        Paths are yielded in reverse order, so that users can add a services module that will override
        the built-in main service module path (e.g. in case a type defined in the built-in module was redefined)
        """
        for path in reversed(self._extra_services_modules):
            yield path
        yield self._main_services_module
