# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

import multiprocessing


class BaseExtractor(object):

    def __init__(self, names, max_workers=None):
        self._names = names
        self._max_workers = max_workers or (multiprocessing.cpu_count() * 4)

    def run(self, job):
        try:
            self.extract(job)
            self.wait_complete()
        except KeyboardInterrupt:
            print('** Shutting down ...')
            self.shutdown()
        else:
            print('^.^ Extracting all packages done!')
        finally:
            self.final()

    def extract(self, job):
        raise NotImplemented

    def wait_complete(self):
        raise NotImplemented

    def shutdown(self):
        raise NotImplemented

    def final(self):
        pass
