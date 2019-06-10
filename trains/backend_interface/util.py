import getpass
import re
from _socket import gethostname
from datetime import datetime

from ..backend_api.services import projects
from ..debugging.log import get_logger


def make_message(s, **kwargs):
    args = dict(
        user=getpass.getuser(),
        host=gethostname(),
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


def get_single_result(entity, query, results, log=None, show_results=10, raise_on_error=True):
    if not results:
        if not raise_on_error:
            return None

        raise ValueError('No {entity}s found when searching for `{query}`'.format(**locals()))

    if not log:
        log = get_logger()

    if len(results) > 1:
        log.warn('More than one {entity} found when searching for `{query}`'
                 ' (showing first {show_results} {entity}s follow)'.format(**locals()))
        for obj in (o if isinstance(o, dict) else o.to_dict() for o in results[:show_results]):
            log.warn('Found {entity} `{obj[name]}` (id={obj[id]})'.format(**locals()))

        if raise_on_error:
            raise ValueError('More than one {entity}s found when searching for ``{query}`'.format(**locals()))

    return results[0]


def at_least_one(_exception_cls=Exception, **kwargs):
    actual = [k for k, v in kwargs.items() if v]
    if len(actual) < 1:
        raise _exception_cls('At least one of (%s) is required' % ', '.join(kwargs.keys()))


def mutually_exclusive(_exception_cls=Exception, _require_at_least_one=True, **kwargs):
    """ Helper for checking mutually exclusive options """
    actual = [k for k, v in kwargs.items() if v]
    if _require_at_least_one:
        at_least_one(_exception_cls=_exception_cls, **kwargs)
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
    return '^%s$' % re.escape(name)

