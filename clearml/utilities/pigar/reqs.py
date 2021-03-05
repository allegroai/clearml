# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

import json
import os
import sys
import six
import fnmatch
import importlib
import ast
import doctest
import collections
import functools
from pathlib2 import Path
from .utils import parse_git_config
from .modules import ImportedModules


def project_import_modules(project_path, ignores):
    """
    copied form pigar req.project_import_modules patching, os.getcwd() is incorrectly used
    """
    modules = ImportedModules()
    try_imports = set()
    local_mods = list()
    ignore_paths = set()
    venv_subdirs = set(('bin', 'etc', 'include', 'lib', 'lib64', 'share'))
    ignore_absolute = []
    if not ignores:
        ignore_paths.add('.git')
    else:
        for path in ignores:
            ignore_paths.add(Path(path).name)

    if os.path.isfile(project_path):
        try:
            fake_path = Path(project_path).name
            with open(project_path, 'rb') as f:
                fmodules, try_ipts = file_import_modules(fake_path, f.read())
                modules |= fmodules
                try_imports |= try_ipts
        except Exception:
            pass
    else:
        cur_dir = project_path
        for dirpath, dirnames, files in os.walk(os.path.abspath(project_path), followlinks=True):
            # see if we have a parent folder in the ignore list
            if ignore_paths & set(Path(dirpath).relative_to(cur_dir).parts):
                continue
            # check if we are in a subfolder of ignore list
            if any(True for prefix in ignore_absolute if dirpath.startswith(prefix)):
                continue
            # Hack detect if this is a virtual-env folder, if so add it to the uignore list
            if set(dirnames) == venv_subdirs:
                ignore_absolute.append(Path(dirpath).as_posix() + os.sep)
                continue

            py_files = list()
            for fn in files:
                # C extension.
                if fn.endswith('.so'):
                    local_mods.append(fn[:-3])
                # Normal Python file.
                if fn.endswith('.py'):
                    local_mods.append(fn[:-3])
                    py_files.append(fn)
            if '__init__.py' in files:
                local_mods.append(os.path.basename(dirpath))
            for file in py_files:
                try:
                    fpath = os.path.join(dirpath, file)
                    fake_path = Path(fpath).relative_to(cur_dir).as_posix()
                    with open(fpath, 'rb') as f:
                        fmodules, try_ipts = file_import_modules(fake_path, f.read())
                        modules |= fmodules
                        try_imports |= try_ipts
                except Exception:
                    pass

    return modules, try_imports, local_mods


def file_import_modules(fpath, fdata):
    """Get single file all imported modules."""
    modules = ImportedModules()
    str_codes = collections.deque([(fdata, 1)])
    try_imports = set()

    while str_codes:
        str_code, lineno = str_codes.popleft()
        ic = ImportChecker(fpath, lineno)
        try:
            parsed = ast.parse(str_code)
            ic.visit(parsed)
        # Ignore SyntaxError in Python code.
        except SyntaxError:
            pass
        modules |= ic.modules
        str_codes.extend(ic.str_codes)
        try_imports |= ic.try_imports
        del ic

    return modules, try_imports


class ImportChecker(object):

    def __init__(self, fpath, lineno):
        self._fpath = fpath
        self._lineno = lineno - 1
        self._modules = ImportedModules()
        self._str_codes = collections.deque()
        self._try_imports = set()

    def visit_Import(self, node, try_=False):
        """As we know: `import a [as b]`."""
        lineno = node.lineno + self._lineno
        for alias in node.names:
            self._modules.add(alias.name, self._fpath, lineno)
            if try_:
                self._try_imports.add(alias.name)

    def visit_ImportFrom(self, node, try_=False):
        """
        As we know: `from a import b [as c]`. If node.level is not 0,
        import statement like this `from .a import b`.
        """
        mod_name = node.module
        level = node.level
        if mod_name is None:
            level -= 1
            mod_name = ''
        for alias in node.names:
            name = level * '.' + mod_name + '.' + alias.name
            self._modules.add(name, self._fpath, node.lineno + self._lineno)
            if try_:
                self._try_imports.add(name)

    def visit_TryExcept(self, node):
        """
        If modules which imported by `try except` and not found,
        maybe them come from other Python version.
        """
        for ipt in node.body:
            if ipt.__class__.__name__.startswith('Import'):
                method = 'visit_' + ipt.__class__.__name__
                getattr(self, method)(ipt, True)
        for handler in node.handlers:
            for ipt in handler.body:
                if ipt.__class__.__name__.startswith('Import'):
                    method = 'visit_' + ipt.__class__.__name__
                    getattr(self, method)(ipt, True)

    # For Python 3.3+
    visit_Try = visit_TryExcept

    def visit_Exec(self, node):
        """
        Check `expression` of `exec(expression[, globals[, locals]])`.
        **Just available in python 2.**
        """
        if hasattr(node.body, 's'):
            self._str_codes.append((node.body.s, node.lineno + self._lineno))
        # PR#13: https://github.com/damnever/pigar/pull/13
        # Sometimes exec statement may be called with tuple in Py2.7.6
        elif hasattr(node.body, 'elts') and len(node.body.elts) >= 1:
            self._str_codes.append(
                (node.body.elts[0].s, node.lineno + self._lineno))

    def visit_Expr(self, node):
        """
        Check `expression` of `eval(expression[, globals[, locals]])`.
        Check `expression` of `exec(expression[, globals[, locals]])`
        in python 3.
        Check `name` of `__import__(name[, globals[, locals[,
        fromlist[, level]]]])`.
        Check `name` or `package` of `importlib.import_module(name,
        package=None)`.
        """
        # Built-in functions
        value = node.value
        if isinstance(value, ast.Call):
            if hasattr(value.func, 'id'):
                if (value.func.id == 'eval' and
                        hasattr(node.value.args[0], 's')):
                    self._str_codes.append(
                        (node.value.args[0].s, node.lineno + self._lineno))
                # **`exec` function in Python 3.**
                elif (value.func.id == 'exec' and
                        hasattr(node.value.args[0], 's')):
                    self._str_codes.append(
                        (node.value.args[0].s, node.lineno + self._lineno))
                # `__import__` function.
                elif (value.func.id == '__import__' and
                        len(node.value.args) > 0 and
                        hasattr(node.value.args[0], 's')):
                    self._modules.add(node.value.args[0].s, self._fpath,
                                      node.lineno + self._lineno)
            # `import_module` function.
            elif getattr(value.func, 'attr', '') == 'import_module':
                module = getattr(value.func, 'value', None)
                if (module is not None and
                        getattr(module, 'id', '') == 'importlib'):
                    args = node.value.args
                    arg_len = len(args)
                    if arg_len > 0 and hasattr(args[0], 's'):
                        name = args[0].s
                        if not name.startswith('.'):
                            self._modules.add(name, self._fpath,
                                              node.lineno + self._lineno)
                        elif arg_len == 2 and hasattr(args[1], 's'):
                            self._modules.add(args[1].s, self._fpath,
                                              node.lineno + self._lineno)

    def visit_FunctionDef(self, node):
        """
        Check docstring of function, if docstring is used for doctest.
        """
        docstring = self._parse_docstring(node)
        if docstring:
            self._str_codes.append((docstring, node.lineno + self._lineno + 2))

    def visit_ClassDef(self, node):
        """
        Check docstring of class, if docstring is used for doctest.
        """
        docstring = self._parse_docstring(node)
        if docstring:
            self._str_codes.append((docstring, node.lineno + self._lineno + 2))

    def visit(self, node):
        """Visit a node, no recursively."""
        for node in ast.walk(node):
            method = 'visit_' + node.__class__.__name__
            getattr(self, method, lambda x: x)(node)

    @staticmethod
    def _parse_docstring(node):
        """Extract code from docstring."""
        docstring = ast.get_docstring(node)
        if docstring:
            parser = doctest.DocTestParser()
            try:
                dt = parser.get_doctest(docstring, {}, None, None, None)
            except ValueError:
                # >>> 'abc'
                pass
            else:
                examples = dt.examples
                return '\n'.join([example.source for example in examples])
        return None

    @property
    def modules(self):
        return self._modules

    @property
    def str_codes(self):
        return self._str_codes

    @property
    def try_imports(self):
        return set((name.split('.')[0] if name and '.' in name else name)
                   for name in self._try_imports)


def _checked_cache(func):
    checked = dict()

    @functools.wraps(func)
    def _wrapper(name):
        if name not in checked:
            checked[name] = func(name)
        return checked[name]

    return _wrapper


@_checked_cache
def is_std_or_local_lib(name):
    """Check whether it is stdlib module.
    True if std lib
    False if installed package
    str if local library
    """
    exist = True
    if six.PY2:
        import imp  # noqa
        from types import FileType  # noqa
        module_info = ('', '', '')
        try:
            module_info = imp.find_module(name)
        except ImportError:
            try:
                # __import__(name)
                importlib.import_module(name)
                module_info = imp.find_module(name)
                sys.modules.pop(name)
            except ImportError:
                exist = False
        # Testcase: ResourceWarning
        if isinstance(module_info[0], FileType):
            module_info[0].close()  # noqa
        mpath = module_info[1]  # noqa
    else:
        module_info = None
        try:
            module_info = importlib.util.find_spec(name) # noqa
        except ImportError:
            return False
        except ValueError:
            # if we got here, the loader failed on us, meaning this is definitely a module and not std
            return False
        if not module_info:
            return False
        mpath = module_info.origin
        # this is std
        if mpath == 'built-in':
            mpath = None

    if exist and mpath is not None:
        if ('site-packages' in mpath or
                'dist-packages' in mpath or
                'bin/' in mpath and mpath.endswith('.py')):
            exist = False
        elif ((sys.prefix not in mpath) and
              (six.PY2 or (sys.base_exec_prefix not in mpath)) and
              (six.PY2 or (sys.base_prefix not in mpath))):
            exist = mpath

    return exist


def get_installed_pkgs_detail():
    """
    HACK: bugfix of the original pigar get_installed_pkgs_detail

    Get mapping for import top level name
    and install package name with version.
    """
    mapping = dict()

    for path in sys.path:
        if os.path.isdir(path) and path.rstrip('/').endswith(
                ('site-packages', 'dist-packages')):
            new_mapping = _search_path(path)
            # BUGFIX:
            # override with previous, just like python resolves imports, the first match is the one used.
            # unlike the original implementation, where the last one is used.
            new_mapping.update(mapping)
            mapping = new_mapping

    # HACK: prefer tensorflow_gpu over tensorflow
    if 'tensorflow_gpu' in mapping:
        mapping['tensorflow'] = mapping['tensorflow_gpu']

    return mapping


def is_base_module(module_path):
    python_base = '{}python{}.{}'.format(os.sep, sys.version_info.major, sys.version_info.minor)
    for path in sys.path:
        if os.path.isdir(path) and path.rstrip('/').endswith(
                (python_base, )):
            if not path[-1] == os.sep:
                path += os.sep
            if module_path.startswith(path):
                return True
    return False


def _search_path(path):
    mapping = dict()

    for file in os.listdir(path):
        # Install from PYPI.
        if fnmatch.fnmatch(file, '*-info'):
            pkg_name, version = file.split('-')[:2]
            if version.endswith('dist'):
                version = version.rsplit('.', 1)[0]
            # Issue for ubuntu: sudo pip install xxx
            elif version.endswith('egg'):
                version = version.rsplit('.', 1)[0]

            mapping_pkg_name = pkg_name
            # pep610 support. add support for new pip>=20.1 git reference feature
            git_direct_json = os.path.join(path, file, 'direct_url.json')
            if os.path.isfile(git_direct_json):
                # noinspection PyBroadException
                try:
                    with open(git_direct_json, 'r') as f:
                        vcs_info = json.load(f)

                    if 'vcs_info' in vcs_info:
                        git_url = '{vcs}+{url}@{commit}#egg={package}'.format(
                            vcs=vcs_info['vcs_info']['vcs'], url=vcs_info['url'],
                            commit=vcs_info['vcs_info']['commit_id'], package=pkg_name)
                        # Bugfix: package name should be the URL link, because we need it unique
                        # mapping[pkg_name] = ('-e', git_url)
                        pkg_name, version = '-e {}'.format(git_url), ''
                    elif 'url' in vcs_info:
                        url_link = vcs_info.get('url', '').strip().lower()
                        if url_link and not url_link.startswith('file://'):
                            pkg_name, version = vcs_info['url'], ''

                except Exception:
                    pass

            # default
            mapping[mapping_pkg_name] = (pkg_name, version)

            # analyze 'top_level.txt' if it exists
            top_level = os.path.join(path, file, 'top_level.txt')
            if not os.path.isfile(top_level):
                continue
            with open(top_level, 'r') as f:
                for line in f:
                    mapping[line.strip()] = (pkg_name, version)

        # Install from local and available in GitHub.
        elif fnmatch.fnmatch(file, '*-link'):
            link = os.path.join(path, file)
            if not os.path.isfile(link):
                continue
            # Link path.
            with open(link, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line != '.':
                        dev_dir = line
            if not dev_dir:
                continue
            if not os.path.exists(dev_dir):
                continue
            # Egg info path.
            info_dir = [_file for _file in os.listdir(dev_dir)
                        if _file.endswith('egg-info')]
            if not info_dir:
                continue
            info_dir = info_dir[0]
            top_level = os.path.join(dev_dir, info_dir, 'top_level.txt')
            # Check whether it can be imported.
            if not os.path.isfile(top_level):
                continue

            # Check .git dir.
            git_path = os.path.join(dev_dir, '.git')
            if os.path.isdir(git_path):
                config = parse_git_config(git_path)
                url = config.get('remote "origin"', {}).get('url')
                if not url:
                    continue
                branch = 'branch "master"'
                if branch not in config:
                    for section in config:
                        if 'branch' in section:
                            branch = section
                            break
                if not branch:
                    continue
                branch = branch.split()[1][1:-1]

                pkg_name = info_dir.split('.egg')[0]
                git_url = 'git+{0}@{1}#egg={2}'.format(url, branch, pkg_name)
                with open(top_level, 'r') as f:
                    for line in f:
                        # Bugfix: package name should be the URL link, because we need it unique
                        # mapping[line.strip()] = ('-e', git_url)
                        mapping[line.strip()] = ('-e {}'.format(git_url), '')

    return mapping
