import sys
from collections import defaultdict
import six

if six.PY2:
    # python2.x
    import __builtin__ as builtins
else:
    # python3.x
    import builtins


class PostImportHookPatching(object):
    _patched = False
    _post_import_hooks = defaultdict(list)

    @staticmethod
    def _init_hook():
        if PostImportHookPatching._patched:
            return
        PostImportHookPatching._patched = True

        if six.PY2:
            # python2.x
            builtins.__org_import__ = builtins.__import__
            builtins.__import__ = PostImportHookPatching._patched_import2
        else:
            # python3.x
            builtins.__org_import__ = builtins.__import__
            builtins.__import__ = PostImportHookPatching._patched_import3

    @staticmethod
    def _patched_import2(name, globals={}, locals={}, fromlist=[], level=-1):
        already_imported = name in sys.modules
        mod = builtins.__org_import__(
            name,
            globals=globals,
            locals=locals,
            fromlist=fromlist,
            level=level)

        if not already_imported and name in PostImportHookPatching._post_import_hooks:
            for hook in PostImportHookPatching._post_import_hooks[name]:
                hook()
        return mod

    @staticmethod
    def _patched_import3(name, globals=None, locals=None, fromlist=(), level=0):
        already_imported = name in sys.modules
        mod = builtins.__org_import__(
            name,
            globals=globals,
            locals=locals,
            fromlist=fromlist,
            level=level)

        if not already_imported and name in PostImportHookPatching._post_import_hooks:
            for hook in PostImportHookPatching._post_import_hooks[name]:
                hook()
        return mod

    @staticmethod
    def add_on_import(name, func):
        PostImportHookPatching._init_hook()
        if not name in PostImportHookPatching._post_import_hooks or \
                func not in PostImportHookPatching._post_import_hooks[name]:
            PostImportHookPatching._post_import_hooks[name].append(func)

    @staticmethod
    def remove_on_import(name, func):
        if name in PostImportHookPatching._post_import_hooks and func in PostImportHookPatching._post_import_hooks[name]:
            PostImportHookPatching._post_import_hooks[name].remove(func)

