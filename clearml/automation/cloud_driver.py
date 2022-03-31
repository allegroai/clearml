import logging
from abc import ABC, abstractmethod
from os import environ

import attr

from ..backend_api import Session
from ..backend_api.session.defs import ENV_AUTH_TOKEN

env_git_user = 'CLEARML_AUTOSCALER_GIT_USER'
env_git_pass = 'CLEARML_AUTOSCALER_GIT_PASSWORD'

bash_script_template = '''\
#!/bin/bash

set -x

apt-get update
apt-get install -y \
        build-essential \
        gcc \
        git \
        python3-dev \
        python3-pip
python3 -m pip install -U pip
python3 -m pip install virtualenv
python3 -m virtualenv clearml_agent_venv
source clearml_agent_venv/bin/activate
python -m pip install clearml-agent
cat << EOF >> ~/clearml.conf
{clearml_conf}
EOF
export CLEARML_API_HOST={api_server}
export CLEARML_WEB_HOST={web_server}
export CLEARML_FILES_HOST={files_server}
export DYNAMIC_INSTANCE_ID=$({instance_id_command})
export CLEARML_WORKER_ID={worker_prefix}:$DYNAMIC_INSTANCE_ID
export CLEARML_API_ACCESS_KEY='{access_key}'
export CLEARML_API_SECRET_KEY='{secret_key}'
export CLEARML_AUTH_TOKEN='{auth_token}'
source ~/.bashrc
{bash_script}
{driver_extra}
python -m clearml_agent --config-file ~/clearml.conf daemon --queue '{queue}' {docker}

if [[ $? -ne 0 ]]
then
  exit 1
fi

shutdown
'''

clearml_conf_template = '''\
agent.git_user="{git_user}"
agent.git_pass="{git_pass}"
{extra_clearml_conf}
'''


@attr.s
class CloudDriver(ABC):
    # git
    git_user = attr.ib()
    git_pass = attr.ib()

    # clearml
    extra_clearml_conf = attr.ib()
    api_server = attr.ib()
    web_server = attr.ib()
    files_server = attr.ib()
    access_key = attr.ib()
    secret_key = attr.ib()
    auth_token = attr.ib()

    # Other
    extra_vm_bash_script = attr.ib()
    docker_image = attr.ib()
    tags = attr.ib(default='')
    session = attr.ib(default=None)

    def __attrs_post_init__(self):
        if self.session is None:
            self.session = Session()

    @abstractmethod
    def spin_up_worker(self, resource, worker_prefix, queue_name, task_id):
        """Creates a new worker for clearml.

        First, create an instance in the cloud and install some required packages.
        Then, define clearml-agent environment variables and run clearml-agent for the specified queue.
        NOTE: - Will wait until instance is running
              - This implementation assumes the instance image already has docker installed

        :param dict resource: resource configuration, as defined in BUDGET and QUEUES.
        :param str worker_prefix: worker name without instance_id
        :param str queue_name: clearml queue to listen to
        :param str task_id: Task ID to restart
        """

    @abstractmethod
    def spin_down_worker(self, instance_id):
        """Destroys the cloud instance.

        :param str instance_id: Cloud instance ID to be destroyed (currently, only AWS EC2 is supported)
        """

    @abstractmethod
    def kind(self):
        """Return driver kind (e.g. 'AWS')"""

    @abstractmethod
    def instance_id_command(self):
        """Return a shell command to get instance ID"""

    @abstractmethod
    def instance_type_key(self):
        """Return key in configuration for instance type"""

    def console_log(self, instance_id):
        """Return log for instance"""
        return ""

    def gen_user_data(self, worker_prefix, queue_name, task_id, cpu_only=False):
        return bash_script_template.format(
            queue=queue_name,
            worker_prefix=worker_prefix,

            auth_token=self.auth_token or '',
            access_key=self.access_key or '',
            api_server=self.api_server,
            clearml_conf=self.clearml_conf(),
            files_server=self.files_server,
            secret_key=self.secret_key or '',
            web_server=self.web_server,

            bash_script=("export NVIDIA_VISIBLE_DEVICES=none; " if cpu_only else "") + self.extra_vm_bash_script,
            driver_extra=self.driver_bash_extra(task_id),
            docker="--docker '{}'".format(self.docker_image) if self.docker_image else "",
            instance_id_command=self.instance_id_command(),
        )

    def clearml_conf(self):
        # TODO: This need to be documented somewhere
        git_user = environ.get(env_git_user) or self.git_user or ''
        git_pass = environ.get(env_git_pass) or self.git_pass or ''

        return clearml_conf_template.format(
            git_user=git_user,
            git_pass=git_pass,
            extra_clearml_conf=self.extra_clearml_conf,
        )

    def driver_bash_extra(self, task_id):
        if not task_id:
            return ''
        return 'python -m clearml_agent --config-file ~/clearml.conf execute --id {}'.format(task_id)

    @classmethod
    def from_config(cls, config):
        session = Session()
        hyper_params, configurations = config['hyper_params'], config['configurations']
        opts = {
            'git_user': hyper_params['git_user'],
            'git_pass': hyper_params['git_pass'],
            'extra_clearml_conf': configurations['extra_clearml_conf'],
            'api_server': session.get_api_server_host(),
            'web_server': session.get_app_server_host(),
            'files_server': session.get_files_server_host(),
            'access_key': session.access_key,
            'secret_key': session.secret_key,
            'auth_token': ENV_AUTH_TOKEN.get(),
            'extra_vm_bash_script': configurations['extra_vm_bash_script'],
            'docker_image': hyper_params['default_docker_image'],
            'tags': hyper_params.get('tags', ''),
            'session': session,
        }
        return cls(**opts)

    def set_scaler(self, scaler):
        self.scaler = scaler

    @property
    def logger(self):
        if self.scaler:
            return self.scaler.logger
        return logging.getLogger('AWSDriver')


def parse_tags(s):
    """
    >>> parse_tags('k1=v1, k2=v2')
    [('k1', 'v1'), ('k2', 'v2')]
    """
    s = s.strip()
    if not s:
        return []

    tags = []
    for kv in s.split(','):
        if '=' not in kv:
            raise ValueError(kv)
        key, value = [v.strip() for v in kv.split('=', 1)]
        if not key or not value:
            raise ValueError(kv)
        tags.append((key, value))
    return tags
