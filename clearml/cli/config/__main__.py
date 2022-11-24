""" ClearML configuration wizard"""
from __future__ import print_function

import argparse
import os

from pathlib2 import Path
from six.moves import input
from six.moves.urllib.parse import urlparse

from clearml.backend_api.session import Session
from clearml.backend_api.session.defs import ENV_HOST
from clearml.backend_config.defs import LOCAL_CONFIG_FILES, LOCAL_CONFIG_FILE_OVERRIDE_VAR
from clearml.config import config_obj
from clearml.utilities.pyhocon import ConfigFactory, ConfigMissingException

description = "\n" \
    "Please create new clearml credentials through the settings page in " \
    "your `clearml-server` web app (e.g. http://localhost:8080//settings/workspace-configuration) \n"\
    "Or create a free account at https://app.clear.ml/settings/workspace-configuration\n\n" \
    "In settings page, press \"Create new credentials\", then press \"Copy to clipboard\".\n" \
    "\n" \
    "Paste copied configuration here:\n"

host_description = """
Editing configuration file: {CONFIG_FILE}
Enter the url of the clearml-server's Web service, for example: {HOST}
"""

# noinspection PyBroadException
try:
    def_host = ENV_HOST.get(default=config_obj.get("api.web_server")) or 'http://localhost:8080'
except Exception:
    def_host = 'http://localhost:8080'


def validate_file(string):
    if not string:
        raise argparse.ArgumentTypeError("expected a valid file path")
    return string


def main():
    default_config_file = LOCAL_CONFIG_FILE_OVERRIDE_VAR.get()
    if not default_config_file:
        for f in LOCAL_CONFIG_FILES:
            default_config_file = f
            if os.path.exists(os.path.expanduser(os.path.expandvars(f))):
                break

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--file", "-F", help="Target configuration file path (default is %(default)s)",
        default=default_config_file,
        type=validate_file
    )

    args = p.parse_args()

    print('ClearML SDK setup process')

    conf_file = Path(os.path.expanduser(args.file)).absolute()
    if conf_file.exists() and conf_file.is_file() and conf_file.stat().st_size > 0:
        print('Configuration file already exists: {}'.format(str(conf_file)))
        print('Leaving setup, feel free to edit the configuration file.')
        return
    print(description, end='')
    sentinel = ''
    parse_input = ''

    if os.environ.get("JPY_PARENT_PID"):
        # When running from a colab instance and calling clearml-init
        # colab will squish the api credentials into a single line
        # The regex splits this single line based on 2 spaces or more
        import re
        api_input = input()
        parse_input = '\n'.join(re.split(r" {2,}", api_input))
    else:
        for line in iter(input, sentinel):
            parse_input += line+'\n'
            if line.rstrip() == '}':
                break

    credentials = None
    api_server = None
    web_server = None
    # noinspection PyBroadException
    try:
        parsed = ConfigFactory.parse_string(parse_input)
        if parsed:
            # Take the credentials in raw form or from api section
            credentials = get_parsed_field(parsed, ["credentials"])
            api_server = get_parsed_field(parsed, ["api_server", "host"])
            web_server = get_parsed_field(parsed, ["web_server"])
    except Exception:
        credentials = credentials or None
        api_server = api_server or None
        web_server = web_server or None

    while not credentials or set(credentials) != {"access_key", "secret_key"}:
        print('Could not parse credentials, please try entering them manually.')
        credentials = read_manual_credentials()

    print('Detected credentials key=\"{}\" secret=\"{}\"'.format(credentials['access_key'],
                                                                 credentials['secret_key'][0:4] + "***"))
    web_input = True
    if web_server:
        host = web_server
    elif api_server:
        web_input = False
        host = api_server
    else:
        print(host_description.format(CONFIG_FILE=args.file, HOST=def_host,))
        host = input_url('WEB Host', '')

    parsed_host = verify_url(host)
    api_host, files_host, web_host = parse_host(parsed_host, allow_input=True)

    # on of these two we configured
    if not web_input:
        web_host = input_url('Web Application Host', web_host)
    else:
        if web_input is True and not web_host:
            web_host = host

    print('\nClearML Hosts configuration:\nWeb App: {}\nAPI: {}\nFile Store: {}\n'.format(
        web_host, api_host, files_host))

    retry = 1
    max_retries = 2
    while retry <= max_retries:  # Up to 2 tries by the user
        if verify_credentials(api_host, credentials):
            break
        retry += 1
        if retry < max_retries + 1:
            credentials = read_manual_credentials()
    else:
        print('Exiting setup without creating configuration file')
        return

    # noinspection PyBroadException
    try:
        default_sdk_conf = Path(__file__).absolute().parents[2] / 'config/default/sdk.conf'
        with open(str(default_sdk_conf), 'rt') as f:
            default_sdk = f.read()
    except Exception:
        print('Error! Could not read default configuration file')
        return
    # noinspection PyBroadException
    try:
        with open(str(conf_file), 'wt') as f:
            header = '# ClearML SDK configuration file\n' \
                     'api {\n' \
                     '    # Notice: \'host\' is the api server (default port 8008), not the web server.\n' \
                     '    api_server: %s\n' \
                     '    web_server: %s\n' \
                     '    files_server: %s\n' \
                     '    # Credentials are generated using the webapp, %s/settings\n' \
                     '    # Override with os environment: CLEARML_API_ACCESS_KEY / CLEARML_API_SECRET_KEY\n' \
                     '    credentials {"access_key": "%s", "secret_key": "%s"}\n' \
                     '}\n' \
                     'sdk ' % (api_host, web_host, files_host,
                               web_host, credentials['access_key'], credentials['secret_key'])
            f.write(header)
            f.write(default_sdk)
    except Exception:
        print('Error! Could not write configuration file at: {}'.format(str(conf_file)))
        return

    print('\nNew configuration stored in {}'.format(str(conf_file)))
    print('ClearML setup completed successfully.')


def parse_host(parsed_host, allow_input=True):
    if parsed_host.netloc.startswith('demoapp.'):
        # this is our demo server
        api_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('demoapp.', 'demoapi.', 1) + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        files_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('demoapp.', 'demofiles.',
                                                                             1) + parsed_host.path
    elif parsed_host.netloc.startswith('app.'):
        # this is our application server
        api_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('app.', 'api.', 1) + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        files_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('app.', 'files.', 1) + parsed_host.path
    elif parsed_host.netloc.startswith('demoapi.'):
        print('{} is the api server, we need the web server. Replacing \'demoapi.\' with \'demoapp.\''.format(
            parsed_host.netloc))
        api_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('demoapi.', 'demoapp.', 1) + parsed_host.path
        files_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('demoapi.', 'demofiles.',
                                                                             1) + parsed_host.path
    elif parsed_host.netloc.startswith('api.'):
        print('{} is the api server, we need the web server. Replacing \'api.\' with \'app.\''.format(
            parsed_host.netloc))
        api_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('api.', 'app.', 1) + parsed_host.path
        files_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('api.', 'files.', 1) + parsed_host.path
    elif parsed_host.port == 8008:
        print('Port 8008 is the api port. Replacing 8080 with 8008 for Web application')
        api_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc.replace(':8008', ':8080', 1) + parsed_host.path
        files_host = parsed_host.scheme + "://" + parsed_host.netloc.replace(':8008', ':8081', 1) + parsed_host.path
    elif parsed_host.port == 8080:
        api_host = parsed_host.scheme + "://" + parsed_host.netloc.replace(':8080', ':8008', 1) + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        files_host = parsed_host.scheme + "://" + parsed_host.netloc.replace(':8080', ':8081', 1) + parsed_host.path
    elif allow_input:
        api_host = ''
        web_host = ''
        files_host = ''
        if not parsed_host.port:
            print('Host port not detected, do you wish to use the default 8080 port n/[y]? ', end='')
            replace_port = input().lower()
            if not replace_port or replace_port == 'y' or replace_port == 'yes':
                api_host = parsed_host.scheme + "://" + parsed_host.netloc + ':8008' + parsed_host.path
                web_host = parsed_host.scheme + "://" + parsed_host.netloc + ':8080' + parsed_host.path
                files_host = parsed_host.scheme + "://" + parsed_host.netloc + ':8081' + parsed_host.path
            elif not replace_port or replace_port.lower() == 'n' or replace_port.lower() == 'no':
                web_host = input_host_port("Web", parsed_host)
                api_host = input_host_port("API", parsed_host)
                files_host = input_host_port("Files", parsed_host)
        if not api_host:
            api_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
    else:
        raise ValueError("Could not parse host name")

    return api_host, files_host, web_host


def verify_credentials(api_host, credentials):
    """check if the credentials are valid"""
    # noinspection PyBroadException
    try:
        print('Verifying credentials ...')
        if api_host:
            Session(api_key=credentials['access_key'], secret_key=credentials['secret_key'], host=api_host,
                    http_retries_config={"total": 2})
            print('Credentials verified!')
            return True
        else:
            print("Can't verify credentials")
            return False
    except Exception:
        print('Error: could not verify credentials: key={} secret={}'.format(
            credentials.get('access_key'), credentials.get('secret_key')))
        return False


def get_parsed_field(parsed_config, fields):
    """
    Parsed the value from web profile page, 'copy to clipboard' option
    :param parsed_config: The parsed value from the web ui
    :type parsed_config: Config object
    :param fields: list of values to parse, will parse by the list order
    :type fields: List[str]
    :return: parsed value if found, None else
    """
    try:
        return parsed_config.get("api").get(fields[0])
    except ConfigMissingException:  # fallback - try to parse the field like it was in web older version
        if len(fields) == 1:
            return parsed_config.get(fields[0])
        elif len(fields) == 2:
            return parsed_config.get(fields[1])
        else:
            return None


def read_manual_credentials():
    print('Enter user access key: ', end='')
    access_key = input()
    print('Enter user secret: ', end='')
    secret_key = input()
    return {"access_key": access_key, "secret_key": secret_key}


def input_url(host_type, host=None):
    while True:
        print('{} configured to: {}'.format(host_type, '[{}] '.format(host) if host else ''), end='')
        parse_input = input()
        if host and (not parse_input or parse_input.lower() == 'yes' or parse_input.lower() == 'y'):
            break
        parsed_host = verify_url(parse_input) if parse_input else None
        if parse_input and parsed_host:
            host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
            break
    return host


def input_host_port(host_type, parsed_host):
    print('Enter port for {} host '.format(host_type), end='')
    replace_port = input().lower()
    return parsed_host.scheme + "://" + parsed_host.netloc + (
        ':{}'.format(replace_port) if replace_port else '') + parsed_host.path


def verify_url(parse_input):
    # noinspection PyBroadException
    try:
        if not parse_input.startswith('http://') and not parse_input.startswith('https://'):
            # if we have a specific port, use http prefix, otherwise assume https
            if ':' in parse_input:
                parse_input = 'http://' + parse_input
            else:
                parse_input = 'https://' + parse_input
        parsed_host = urlparse(parse_input)
        if parsed_host.scheme not in ('http', 'https'):
            parsed_host = None
    except Exception:
        parsed_host = None
        print('Could not parse url {}\nEnter your clearml-server host: '.format(parse_input), end='')
    return parsed_host


if __name__ == '__main__':
    main()
