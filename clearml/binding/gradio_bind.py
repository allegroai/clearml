import sys
from logging import getLogger
from .frameworks import _patched_call  # noqa
from .import_bind import PostImportHookPatching
from ..utilities.networking import get_private_ip


class PatchGradio:
    _current_task = None
    __patched = False

    _default_gradio_address = "0.0.0.0"
    _default_gradio_port = 7860
    _root_path_format = "/service/{}"
    __server_config_warning = set()

    @classmethod
    def update_current_task(cls, task=None):
        cls._current_task = task
        if cls.__patched:
            return
        if "gradio" in sys.modules:
            cls.patch_gradio()
        else:
            PostImportHookPatching.add_on_import("gradio", cls.patch_gradio)

    @classmethod
    def patch_gradio(cls):
        if cls.__patched:
            return
        # noinspection PyBroadException
        try:
            import gradio

            gradio.networking.start_server = _patched_call(
                gradio.networking.start_server, PatchGradio._patched_start_server
            )
            gradio.routes.App.__init__ = _patched_call(gradio.routes.App.__init__, PatchGradio._patched_init)
        except Exception:
            pass
        cls.__patched = True

    @staticmethod
    def _patched_start_server(original_fn, self, server_name=None, server_port=None, *args, **kwargs):
        if not PatchGradio._current_task:
            return original_fn(self, server_name, server_port, *args, **kwargs)
        PatchGradio._current_task._set_runtime_properties(
            {"_SERVICE": "EXTERNAL", "_ADDRESS": get_private_ip(), "_PORT": PatchGradio._default_gradio_port}
        )
        PatchGradio._current_task.set_system_tags(["external_service"])
        PatchGradio.__warn_on_server_config(server_name, server_port)
        server_name = PatchGradio._default_gradio_address
        server_port = PatchGradio._default_gradio_port
        return original_fn(self, server_name, server_port, *args, **kwargs)

    @staticmethod
    def _patched_init(original_fn, *args, **kwargs):
        if not PatchGradio._current_task:
            return original_fn(*args, **kwargs)
        PatchGradio.__warn_on_server_config(kwargs.get("server_name"), kwargs.get("server_port"))
        kwargs["root_path"] = PatchGradio._root_path_format.format(PatchGradio._current_task.id)
        kwargs["root_path_in_servers"] = False
        kwargs["server_name"] = PatchGradio._default_gradio_address
        kwargs["server_port"] = PatchGradio._default_gradio_port
        return original_fn(*args, **kwargs)

    @classmethod
    def __warn_on_server_config(cls, server_name, server_port):
        if server_name is None and server_port is None:
            return
        if server_name is not None and server_port is not None:
            server_config = "{}:{}".format(server_name, server_port)
            what_to_ignore = "name and port"
        elif server_name is not None:
            server_config = str(server_name)
            what_to_ignore = "name"
        else:
            server_config = str(server_port)
            what_to_ignore = "port"
        if server_config in cls.__server_config_warning:
            return
        cls.__server_config_warning.add(server_config)
        getLogger().warning(
            "ClearML only supports '{}:{}'as the Gradio server. Ignoring {} '{}'".format(
                PatchGradio._default_gradio_address, PatchGradio._default_gradio_port, what_to_ignore, server_config
            )
        )
