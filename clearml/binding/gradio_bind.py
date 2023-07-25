import sys
from logging import getLogger
from .frameworks import _patched_call  # noqa
from .import_bind import PostImportHookPatching
from ..utilities.networking import get_private_ip
from ..config import running_remotely


class PatchGradio:
    _current_task = None
    __patched = False

    _default_gradio_address = "0.0.0.0"
    _default_gradio_port = 7860
    _root_path_format = "/service/{}/"
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

            gradio.routes.App.get_blocks = _patched_call(gradio.routes.App.get_blocks, PatchGradio._patched_get_blocks)
            gradio.blocks.Blocks.launch = _patched_call(gradio.blocks.Blocks.launch, PatchGradio._patched_launch)
        except Exception:
            pass
        cls.__patched = True

    @staticmethod
    def _patched_get_blocks(original_fn, *args, **kwargs):
        blocks = original_fn(*args, **kwargs)
        if not PatchGradio._current_task or not running_remotely():
            return blocks
        blocks.config["root"] = PatchGradio._root_path_format.format(PatchGradio._current_task.id)
        blocks.root = blocks.config["root"]
        return blocks

    @staticmethod
    def _patched_launch(original_fn, *args, **kwargs):
        if not PatchGradio._current_task:
            return original_fn(*args, **kwargs)
        PatchGradio.__warn_on_server_config(
            kwargs.get("server_name"),
            kwargs.get("server_port"),
            kwargs.get("root_path")
        )
        if not running_remotely():
            return original_fn(*args, **kwargs)
        # noinspection PyProtectedMember
        PatchGradio._current_task._set_runtime_properties(
            {"_SERVICE": "EXTERNAL", "_ADDRESS": get_private_ip(), "_PORT": PatchGradio._default_gradio_port}
        )
        PatchGradio._current_task.set_system_tags(["external_service"])
        kwargs["server_name"] = PatchGradio._default_gradio_address
        kwargs["server_port"] = PatchGradio._default_gradio_port
        kwargs["root_path"] = PatchGradio._root_path_format.format(PatchGradio._current_task.id)
        # noinspection PyBroadException
        try:
            return original_fn(*args, **kwargs)
        except Exception:
            del kwargs["root_path"]
            return original_fn(*args, **kwargs)

    @classmethod
    def __warn_on_server_config(cls, server_name, server_port, root_path):
        if (server_name is None or server_name == PatchGradio._default_gradio_address) and \
                (server_port is None and server_port == PatchGradio._default_gradio_port):
            return
        if (server_name, server_port, root_path) in cls.__server_config_warning:
            return
        cls.__server_config_warning.add((server_name, server_port, root_path))
        if server_name is not None and server_port is not None:
            server_config = "{}:{}".format(server_name, server_port)
            what_to_ignore = "name and port"
        elif server_name is not None:
            server_config = str(server_name)
            what_to_ignore = "name"
        else:
            server_config = str(server_port)
            what_to_ignore = "port"
        getLogger().warning(
            "ClearML only supports '{}:{}' as the Gradio server. Ignoring {} '{}' in remote execution".format(
                PatchGradio._default_gradio_address, PatchGradio._default_gradio_port, what_to_ignore, server_config
            )
        )
        if root_path is not None:
            getLogger().warning(
                "ClearML will override root_path '{}' to '{}' in remote execution".format(
                    root_path, PatchGradio._root_path_format.format(PatchGradio._current_task.id)
                )
            )
