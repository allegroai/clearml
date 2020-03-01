# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

import collections


# FIXME: Just a workaround, not a radical cure..
_special_cases = {
    "dogpile.cache": "dogpile.cache",
    "dogpile.core": "dogpile.core",
    "ruamel.yaml": "ruamel.yaml",
    "ruamel.ordereddict": "ruamel.ordereddict",
}


class Modules(dict):
    """Modules object will be used to store modules information."""

    def __init__(self):
        super(Modules, self).__init__()


class ImportedModules(Modules):

    def __init__(self):
        super(ImportedModules, self).__init__()

    def add(self, name, file, lineno):
        if name is None:
            return

        names = list()
        special_name = '.'.join(name.split('.')[:2])
        # Flask extension.
        if name.startswith('flask.ext.'):
            names.append('flask')
            names.append('flask_' + name.split('.')[2])
        # Special cases..
        elif special_name in _special_cases:
            names.append(_special_cases[special_name])
        # Other.
        elif '.' in name and not name.startswith('.'):
            names.append(name.split('.')[0])
        else:
            names.append(name)

        for nm in names:
            if nm not in self:
                self[nm] = _Locations()
            self[nm].add(file, lineno)

    def __or__(self, obj):
        for name, locations in obj.items():
            for file, linenos in locations.items():
                for lineno in linenos:
                    self.add(name, file, lineno)
        return self


class ReqsModules(Modules):

    _Detail = collections.namedtuple('Detail', ['version', 'comments'])

    def __init__(self):
        super(ReqsModules, self).__init__()
        self._sorted = None

    def add(self, package, version, locations):
        if package in self:
            self[package].comments.extend(locations)
        else:
            self[package] = self._Detail(version, locations)

    def sorted_items(self):
        if self._sorted is None:
            self._sorted = sorted(self.items())
        return self._sorted

    def remove(self, *names):
        for name in names:
            if name in self:
                self.pop(name)
        self._sorted = None


class _Locations(dict):
    """_Locations store code locations(file, linenos)."""

    def __init__(self):
        super(_Locations, self).__init__()
        self._sorted = None

    def add(self, file, lineno):
        if file in self and lineno not in self[file]:
            self[file].append(lineno)
        else:
            self[file] = [lineno]

    def extend(self, obj):
        for file, linenos in obj.items():
            for lineno in linenos:
                self.add(file, lineno)

    def sorted_items(self):
        if self._sorted is None:
            self._sorted = [
                '{0}: {1}'.format(f, ','.join([str(n) for n in sorted(ls)]))
                for f, ls in sorted(self.items())
            ]
        return self._sorted
