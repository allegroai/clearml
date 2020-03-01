# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import


from ..utils import PY32


if PY32:
    from .thread_extractor import ThreadExtractor as Extractor
else:
    from .gevent_extractor import GeventExtractor as Extractor
