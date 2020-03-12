from six.moves.urllib.parse import quote, urlparse, urlunparse
import six
import fnmatch


def get_config_object_matcher(**patterns):
    unsupported = {k: v for k, v in patterns.items() if not isinstance(v, six.string_types)}
    if unsupported:
        raise ValueError('Unsupported object matcher (expecting string): %s'
                         % ', '.join('%s=%s' % (k, v) for k, v in unsupported.items()))

    def _matcher(**kwargs):
        for key, value in kwargs.items():
            if not value:
                continue
            pat = patterns.get(key)
            if pat and fnmatch.fnmatch(value, pat):
                return True
    return _matcher


def quote_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        return url
    parsed = parsed._replace(path=quote(parsed.path))
    return urlunparse(parsed)
