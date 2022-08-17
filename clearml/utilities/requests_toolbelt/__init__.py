# -*- coding: utf-8 -*-
"""
requests-toolbelt
=================

See https://toolbelt.readthedocs.io/ for documentation

:copyright: (c) 2014 by Ian Cordasco and Cory Benfield
:license: Apache v2.0, see LICENSE for more details
"""

from .multipart import (
    MultipartEncoder, MultipartEncoderMonitor, MultipartDecoder,
    ImproperBodyPartContentException, NonMultipartContentTypeException
    )

__title__ = 'requests-toolbelt'
__authors__ = 'Ian Cordasco, Cory Benfield'
__license__ = 'Apache v2.0'
__copyright__ = 'Copyright 2014 Ian Cordasco, Cory Benfield'
__version__ = '0.9.1'
__version_info__ = tuple(int(i) for i in __version__.split('.'))

__all__ = [
    'MultipartEncoder', 'MultipartEncoderMonitor',
    'MultipartDecoder', 'ImproperBodyPartContentException',
    'NonMultipartContentTypeException', '__title__', '__authors__',
    '__license__', '__copyright__', '__version__', '__version_info__',
]
