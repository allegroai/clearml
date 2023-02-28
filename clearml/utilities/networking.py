import requests
import socket
import subprocess
from typing import Optional


def get_private_ip():
    # type: () -> str
    """
    Get the private IP of this machine

    :return: A string representing the IP of this machine
    """
    approaches = (
        _get_private_ip_from_socket,
        _get_private_ip_from_subprocess,
    )

    for approach in approaches:
        # noinspection PyBroadException
        try:
            return approach()
        except Exception:
            continue

    raise Exception("error getting private IP")


def get_public_ip():
    # type: () -> Optional[str]
    """
    Get the public IP of this machine. External services such as `https://api.ipify.org` or `https://ident.me`
    are used to get the IP

    :return: A string representing the IP of this machine or `None` if getting the IP failed
    """
    for external_service in ["https://api.ipify.org", "https://ident.me"]:
        ip = get_public_ip_from_external_service(external_service)
        if ip:
            return ip
    return None


def get_public_ip_from_external_service(external_service, timeout=5):
    # type: (str, Optional[int]) -> Optional[str]
    """
    Get the public IP of this machine from an external service.
    Fetching the IP is done via a GET request. The whole content of the request
    should be the IP address

    :param external_service: The address of the extrenal service
    :param timeout: The GET request timeout

    :return: A string representing the IP of this machine or `None` if getting the IP failed
    """
    # noinspection PyBroadException
    try:
        response = requests.get(external_service, timeout=timeout)
        if not response.ok:
            return None
        ip = response.content.decode("utf8")
        # check that we actually received an IP address
        # noinspection PyBroadException
        try:
            socket.inet_pton(socket.AF_INET, ip)
            return ip
        except Exception:
            socket.inet_pton(socket.AF_INET6, ip)
            return ip
    except Exception:
        return None


def _get_private_ip_from_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(("8.8.8.8", 1))
        ip = s.getsockname()[0]
    except Exception as e:
        raise e
    finally:
        s.close()
    return ip


def _get_private_ip_from_subprocess():
    return subprocess.check_output("hostname -I", shell=True).split()[0].decode("utf-8")
