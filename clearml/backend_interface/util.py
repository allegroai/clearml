import getpass
import re
from _socket import gethostname
from datetime import datetime
from typing import Optional, Any

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

from ..backend_api.services import projects, queues
from ..debugging.log import get_logger, LoggerRoot


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


def get_existing_project(session, project_name):
    """Return either the project ID if it exists, an empty string if it doesn't or None if backend request failed."""
    res = session.send(projects.GetAllRequest(
        name=exact_match_regex(project_name), only_fields=['id'], search_hidden=True, _allow_extra_fields_=True))
    if not res:
        return None
    if res.response and res.response.projects:
        return res.response.projects[0].id
    return ""


def rename_project(session, project_name, new_project_name):
    # type: (Any, str, str) -> bool
    """
    Rename a project

    :param session: Session to send the request through
    :param project_name: Name of the project you want to rename
    :param new_project_name: New name for the project

    :return: True if the rename succeded and False otherwise
    """
    project_id = get_existing_project(session, project_name)
    if not project_id:
        return False
    res = session.send(projects.UpdateRequest(project=project_id, name=new_project_name))
    if res and res.response and res.response.updated:
        return True
    return False


def get_or_create_project(session, project_name, description=None, system_tags=None, project_id=None):
    """Return the ID of an existing project, or if it does not exist, make a new one and return that ID instead."""
    project_system_tags = []
    if not project_id:
        res = session.send(projects.GetAllRequest(
            name=exact_match_regex(project_name),
            only_fields=['id', 'system_tags'] if system_tags else ['id'],
            search_hidden=True, _allow_extra_fields_=True))

        if res and res.response and res.response.projects:
            project_id = res.response.projects[0].id
            if system_tags:
                project_system_tags = res.response.projects[0].system_tags

    if project_id and system_tags and (not project_system_tags or
                                       set(project_system_tags) & set(system_tags) != set(system_tags)):
        # set system_tags
        session.send(
            projects.UpdateRequest(
                project=project_id, system_tags=list(set((project_system_tags or []) + system_tags))
            )
        )

    if project_id:
        return project_id

    # Project was not found, so create a new one
    res = session.send(projects.CreateRequest(
        name=project_name, description=description or '', system_tags=system_tags))
    return res.response.id if res else None


def get_queue_id(session, queue):
    # type: ('Session', str) -> Optional[str] # noqa: F821
    if not queue:
        return None

    res = session.send(queues.GetByIdRequest(queue=queue))
    if res and res.response.queue:
        return queue
    res = session.send(queues.GetAllRequest(name=exact_match_regex(queue), only_fields=['id']))
    if res and res.response and res.response.queues:
        if len(res.response.queues) > 1:
            LoggerRoot.get_base_logger().info(
                "Multiple queues with name={}, selecting queue id={}".format(queue, res.response.queues[0].id))
        return res.response.queues[0].id

    return None


def get_num_enqueued_tasks(session, queue_id):
    # type: ('Session', str) -> Optional[int] # noqa: F821
    res = session.send(queues.GetNumEntriesRequest(queue=queue_id))
    if res and res.response and res.response.num is not None:
        return res.response.num
    return None


# Hack for supporting windows
def get_epoch_beginning_of_time(timezone_info=None):
    return datetime(1970, 1, 1).replace(tzinfo=timezone_info if timezone_info else utc_timezone)


def get_single_result(entity, query, results, log=None, show_results=1, raise_on_error=True, sort_by_date=True):
    if not results:
        if not raise_on_error:
            return None

        raise ValueError('No {entity}s found when searching for `{query}`'.format(**locals()))

    if len(results) > 1:
        if show_results:
            if not log:
                log = get_logger()
            if show_results > 1:
                log.warning('{num} {entity} found when searching for `{query}`'
                            ' (showing first {show_results} {entity}s follow)'.format(num=len(results), **locals()))
            else:
                log.warning('{num} {entity} found when searching for `{query}`'.format(num=len(results), **locals()))

        if sort_by_date:
            relative_time = get_epoch_beginning_of_time()
            # sort results based on timestamp and return the newest one
            if hasattr(results[0], 'last_update'):
                results = sorted(results, key=lambda x: int((x.last_update - relative_time).total_seconds()
                                                            if x.last_update else 0), reverse=True)
            elif hasattr(results[0], 'created'):
                results = sorted(results, key=lambda x: int((x.created - relative_time).total_seconds()
                                                            if x.created else 0), reverse=True)

        if show_results and log:
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
    if not all(isinstance(x, key_types) for x in obj.keys()):
        raise ValueError('%skeys must all be strings' % ('%s ' % desc if desc else ''))
    if not all(isinstance(x, value_types) for x in obj.values()):
        raise ValueError('%svalues must all be integers' % ('%s ' % desc if desc else ''))


def exact_match_regex(name):
    """ Convert string to a regex representing an exact match """
    return '^%s$' % re.escape(name or '')


def datetime_to_isoformat(o):
    if isinstance(o, datetime):
        return o.isoformat()
    return None


def datetime_from_isoformat(o):
    # type: (str) -> Optional[datetime]
    if not o:
        return None
    if isinstance(o, datetime):
        return o
    try:
        return datetime.strptime(o.split('+')[0], "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(o.split('+')[0], "%Y-%m-%dT%H:%M:%S")
