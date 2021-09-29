# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

import os
import codecs

import six

from .reqs import project_import_modules, is_std_or_local_lib, is_base_module
from .utils import lines_diff
from .log import logger
from .modules import ReqsModules


class GenerateReqs(object):
    _force_modules_reqs = dict()

    def __init__(self, save_path, project_path, ignores,
                 installed_pkgs, comparison_operator='=='):
        self._save_path = save_path
        self._project_path = project_path
        self._ignores = ignores
        self._installed_pkgs = installed_pkgs
        self._maybe_local_mods = set()
        self._local_mods = dict()
        self._relative_imports = set()
        self._comparison_operator = comparison_operator

    def extract_reqs(self, module_callback=None, entry_point_filename=None):
        """Extract requirements from project."""

        reqs = ReqsModules()
        guess = ReqsModules()
        local = ReqsModules()

        num_local_mod = 0
        if self.__module__:
            # create a copy, do not change the class set
            our_module = self.__module__.split('.')[0]
            if our_module and our_module not in self._force_modules_reqs:
                from ...version import __version__
                self._force_modules_reqs[our_module] = __version__

        # make the entry point absolute (relative to the root path)
        if entry_point_filename and not os.path.isabs(entry_point_filename):
            entry_point_filename = os.path.join(self._project_path, entry_point_filename) \
                if os.path.isdir(self._project_path) else None

        # check if the entry point script is self contained, i.e. does not use the rest of the project
        if entry_point_filename and os.path.isfile(entry_point_filename) and not self._local_mods:
            modules, try_imports, local_mods = project_import_modules(entry_point_filename, self._ignores)
            if not local_mods:
                # update the self._local_mods
                self._filter_modules(modules, local_mods)
                # check how many local modules we have, excluding ourselves
                num_local_mod = len(set(self._local_mods.keys()) - set(self._force_modules_reqs.keys()))

            # if we have any module/package we cannot find, take no chances and scan the entire project
            # if we have local modules and they are not just us.
            if num_local_mod or local_mods or self._relative_imports:
                modules, try_imports, local_mods = project_import_modules(
                    self._project_path, self._ignores)

        else:
            modules, try_imports, local_mods = project_import_modules(
                self._project_path, self._ignores)

        if module_callback:
            modules = module_callback(modules)

        # Filtering modules
        candidates = self._filter_modules(modules, local_mods)

        # make sure we are in candidates
        candidates |= set(self._force_modules_reqs.keys())

        logger.info('Check module in local environment.')
        reqs_module_name = []
        for name in candidates:
            logger.info('Checking module: %s', name)
            if name in self._installed_pkgs:
                pkg_name, version = self._installed_pkgs[name]
                if name not in modules:
                    modules.add(name, name, 0)
                if not version and pkg_name and pkg_name.startswith('-e '):
                    reqs.add('{} @ {}'.format(name, pkg_name.replace('-e ', '', 1)), version, modules[name])
                else:
                    reqs.add(pkg_name, version, modules[name])
                reqs_module_name.append(name)
            elif name in modules:
                guess.add(name, 0, modules[name])

        # add local modules, so we know what is used but not installed.
        project_path = os.path.realpath(self._project_path)
        for name in self._local_mods:
            if name in modules and name not in reqs_module_name:
                if name in self._force_modules_reqs:
                    reqs.add(name, self._force_modules_reqs[name], modules[name])
                    reqs_module_name.append(name)
                    continue

                # if this is a base module, we have it in installed modules but package name is None
                mod_path = os.path.realpath(self._local_mods[name])
                if is_base_module(mod_path):
                    continue

                # if this is a folder of our project, we can safely ignore it
                if (six.PY3 and os.path.commonpath([project_path]) == os.path.commonpath([project_path, mod_path])) or \
                        (six.PY2 and
                         os.path.commonprefix([project_path]) == os.path.commonprefix([project_path, mod_path])):
                    continue

                relpath = os.path.relpath(self._local_mods[name], self._project_path)
                if not relpath.startswith('.'):
                    relpath = '.' + os.path.sep + relpath
                local.add(name, relpath, modules[name])

        return reqs, try_imports, guess, local

    @classmethod
    def get_forced_modules(cls):
        return cls._force_modules_reqs

    @classmethod
    def add_forced_module(cls, module_name, module_version):
        cls._force_modules_reqs[module_name] = module_version

    def _write_reqs(self, reqs):
        print('Writing requirements to "{0}"'.format(
            self._save_path))
        with open(self._save_path, 'w+') as f:
            f.write('# Requirements automatically generated by pigar.\n'
                    '# https://github.com/damnever/pigar\n')
            for k, v in reqs.sorted_items():
                f.write('\n')
                f.write(''.join(['# {0}\n'.format(c)
                                 for c in v.comments.sorted_items()]))
                if k == '-e':
                    f.write('{0} {1}\n'.format(k, v.version))
                elif v:
                    f.write('{0} {1} {2}\n'.format(
                        k, self._comparison_operator, v.version))
                else:
                    f.write('{0}\n'.format(k))

    def _best_matchs(self, name, pkgs):
        # If imported name equals to package name.
        if name in pkgs:
            return [pkgs[pkgs.index(name)]]
        # If not, return all possible packages.
        return pkgs

    def _filter_modules(self, modules, local_mods):
        candidates = set()

        logger.info('Filtering modules ...')
        for module in modules:
            logger.info('Checking module: %s', module)
            if not module:
                continue
            if module.startswith('.'):
                self._relative_imports.add(module)
                continue
            if module in local_mods:
                self._maybe_local_mods.add(module)
            module_std_local = is_std_or_local_lib(module)
            if module_std_local is True:
                continue
            if isinstance(module_std_local, str):
                self._local_mods[module] = module_std_local
                continue
            candidates.add(module)

        return candidates

    def _invalid_reqs(self, reqs):
        for name, detail in reqs.sorted_items():
            print(
                '  {0} referenced from:\n    {1}'.format(
                    name,
                    '\n    '.join(detail.comments.sorted_items())
                )
            )

    def _save_old_reqs(self):
        if os.path.isfile(self._save_path):
            with codecs.open(self._save_path, 'rb', 'utf-8') as f:
                self._old_reqs = f.readlines()

    def _reqs_diff(self):
        if not hasattr(self, '_old_reqs'):
            return
        with codecs.open(self._save_path, 'rb', 'utf-8') as f:
            new_reqs = f.readlines()
        is_diff, diffs = lines_diff(self._old_reqs, new_reqs)
        msg = 'Requirements file has been covered, '
        if is_diff:
            msg += 'there is the difference:'
            print('{0}\n{1}'.format(msg, ''.join(diffs)), end='')
        else:
            msg += 'no difference.'
            print(msg)
