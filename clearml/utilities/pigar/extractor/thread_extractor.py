# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

import concurrent.futures

from .extractor import BaseExtractor

from ..log import logger


class ThreadExtractor(BaseExtractor):
    """Extractor use thread pool execute tasks.

    Can be used to extract /simple/<pkg_name> or /pypi/<pkg_name>/json.

    FIXME: can not deliver SIG_INT to threads in Python 2.
    """

    def __init__(self, names, max_workers=None):
        super(self.__class__, self).__init__(names, max_workers)
        self._futures = dict()

    def extract(self, job):
        """Extract url by package name."""
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_workers) as executor:
            for name in self._names:
                self._futures[executor.submit(job, name)] = name

    def wait_complete(self):
        """Wait for futures complete done."""
        for future in concurrent.futures.as_completed(self._futures.keys()):
            try:
                error = future.exception()
            except concurrent.futures.CancelledError:
                break
            name = self._futures[future]
            if error is not None:
                err_msg = 'Extracting "{0}", got: {1}'.format(name, error)
                logger.error(err_msg)

    def shutdown(self):
        for future in self._futures:
            future.cancel()
