import getpass
import re
from _socket import gethostname
from datetime import datetime
try:
    from datetime import timezone
    utc_timezone = timezone.utc
except ImportError:
    from datetime import tzinfo, timedelta

    class UTC(tzinfo):
        def utcoffset(self, dt):
            return timedelta(0)

        def tzname(self, dt):
            return "UTC"

        def dst(self, dt):
            return timedelta(0)
    utc_timezone = UTC()

from ..backend_api.services import projects
from ..debugging.log import get_logger


def make_message(s, **kwargs):
    # noinspection PyBroadException
    try:
        user = getpass.getuser()
    except Exception:
        # noinspection PyBroadException
        try:
            import os
            user = '{}'.format(os.getuid())
        except Exception:
            user = 'unknown'

    # noinspection PyBroadException
    try:
        host = gethostname()
    except Exception:
        host = 'localhost'

    args = dict(
        user=user,
        host=host,
        time=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    )
    args.update(kwargs)
    return s % args


def get_or_create_project(session, project_name, description=None):
    res = session.send(projects.GetAllRequest(name=exact_match_regex(project_name)))
    if res.response.projects:
        return res.response.projects[0].id
    res = session.send(projects.CreateRequest(name=project_name, description=description))
    return res.response.id


# Hack for supporting windows
def get_epoch_beginning_of_time(timezone_info=None):
    return datetime(1970, 1, 1).replace(tzinfo=timezone_info if timezone_info else utc_timezone)


def get_single_result(entity, query, results, log=None, show_results=10, raise_on_error=True, sort_by_date=True):
    if not results:
        if not raise_on_error:
            return None

        raise ValueError('No {entity}s found when searching for `{query}`'.format(**locals()))

    if not log:
        log = get_logger()

    if len(results) > 1:
        log.warning('More than one {entity} found when searching for `{query}`'
                    ' (showing first {show_results} {entity}s follow)'.format(**locals()))
        if sort_by_date:
            relative_time = get_epoch_beginning_of_time()
            # sort results based on timestamp and return the newest one
            if hasattr(results[0], 'last_update'):
                results = sorted(results, key=lambda x: int((x.last_update - relative_time).total_seconds()
                                                            if x.last_update else 0), reverse=True)
            elif hasattr(results[0], 'created'):
                results = sorted(results, key=lambda x: int((x.created - relative_time).total_seconds()
                                                            if x.created else 0), reverse=True)

        for i, obj in enumerate(o if isinstance(o, dict) else o.to_dict() for o in results[:show_results]):
            selected = 'Selected' if i == 0 else 'Additionally found'
            log.warning('{selected} {entity} `{obj[name]}` (id={obj[id]})'.format(**locals()))

        if raise_on_error:
            raise ValueError('More than one {entity}s found when searching for ``{query}`'.format(**locals()))

    return results[0]


def at_least_one(_exception_cls=Exception, _check_none=False, **kwargs):
    actual = [k for k, v in kwargs.items() if (v is not None if _check_none else v)]
    if len(actual) < 1:
        raise _exception_cls('At least one of (%s) is required' % ', '.join(kwargs.keys()))


def mutually_exclusive(_exception_cls=Exception, _require_at_least_one=True, _check_none=False, **kwargs):
    """ Helper for checking mutually exclusive options """
    actual = [k for k, v in kwargs.items() if (v is not None if _check_none else v)]
    if _require_at_least_one:
        at_least_one(_exception_cls=_exception_cls, _check_none=_check_none, **kwargs)
    if len(actual) > 1:
        raise _exception_cls('Only one of (%s) is allowed' % ', '.join(kwargs.keys()))


def validate_dict(obj, key_types, value_types, desc=''):
    if not isinstance(obj, dict):
        raise ValueError('%sexpected a dictionary' % ('%s: ' % desc if desc else ''))
    if not all(isinstance(l, key_types) for l in obj.keys()):
        raise ValueError('%skeys must all be strings' % ('%s ' % desc if desc else ''))
    if not all(isinstance(l, value_types) for l in obj.values()):
        raise ValueError('%svalues must all be integers' % ('%s ' % desc if desc else ''))


def exact_match_regex(name):
    """ Convert string to a regex representing an exact match """
    return '^%s$' % re.escape(name or '')
