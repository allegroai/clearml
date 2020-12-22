from __future__ import absolute_import, division, print_function

import json
import os

import requests

from ..backend_api.session import Session
from ..backend_config import EnvEntry

from .version import Version


class CheckPackageUpdates(object):
    _package_version_checked = False

    @classmethod
    def check_new_package_available(cls, only_once=False):
        """
        :return: True, if there is a newer package in PyPI.
        """
        if only_once and cls._package_version_checked:
            return None

        # noinspection PyBroadException
        try:
            cls._package_version_checked = True
            client, version = Session._client[0]
            version = Version(version)
            is_demo = 'https://demoapi.demo.clear.ml/'.startswith(Session.get_api_server_host())

            update_server_releases = requests.get(
                'https://updates.clear.ml/updates',
                json={"demo": is_demo,
                      "versions": {c: str(v) for c, v in Session._client},
                      "CI": str(os.environ.get('CI', ''))},
                timeout=3.0
            )

            if update_server_releases.ok:
                update_server_releases = update_server_releases.json()
            else:
                return None

            client_answer = update_server_releases.get(client, {})
            if "version" not in client_answer:
                return None

            # do not output upgrade message if we are running inside a CI process.
            if EnvEntry("CI", type=bool, ignore_errors=True).get():
                return None

            latest_version = Version(client_answer["version"])

            if version >= latest_version:
                return None
            not_patch_upgrade = latest_version.release[:2] == version.release[:2]
            return str(latest_version), not_patch_upgrade, client_answer.get("description").split("\r\n")
        except Exception:
            return None

    @staticmethod
    def get_version_from_updates_server(cur_version):
        """
        Get the latest version for clearml from updates server
        :param cur_version: The current running version of clearml
        :type cur_version: Version
        """
        try:
            _ = requests.get('https://updates.clear.ml/updates',
                             data=json.dumps({"versions": {"clearml": str(cur_version)}}),
                             timeout=1.0)
            return
        except Exception:
            pass
