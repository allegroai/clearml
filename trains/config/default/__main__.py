from __future__ import print_function

from six.moves import input
from pyhocon import ConfigFactory
from pathlib2 import Path
from six.moves.urllib.parse import urlparse

from trains.backend_api.session.defs import ENV_HOST
from trains.backend_config.defs import LOCAL_CONFIG_FILES
from trains.config import config_obj


description = """
Please create new credentials using the web app: {}/profile
In the Admin page, press "Create new credentials", then press "Copy to clipboard"

Paste credentials here: """

try:
    def_host = ENV_HOST.get(default=config_obj.get("api.web_server")) or 'http://localhost:8080'
except Exception:
    def_host = 'http://localhost:8080'

host_description = """
Editing configuration file: {CONFIG_FILE}
Enter the url of the trains-server's Web service, for example: {HOST}
""".format(
    CONFIG_FILE=LOCAL_CONFIG_FILES[0],
    HOST=def_host,
)


def main():
    print('TRAINS SDK setup process')
    conf_file = Path(LOCAL_CONFIG_FILES[0]).absolute()
    if conf_file.exists() and conf_file.is_file() and conf_file.stat().st_size > 0:
        print('Configuration file already exists: {}'.format(str(conf_file)))
        print('Leaving setup, feel free to edit the configuration file.')
        return

    print(host_description)
    web_host = input_url('Web Application Host', '')
    parsed_host = verify_url(web_host)

    if parsed_host.port == 8008:
        print('Port 8008 is the api port. Replacing 8080 with 8008 for Web application')
        api_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc.replace(':8008', ':8080', 1) + parsed_host.path
        files_host = parsed_host.scheme + "://" + parsed_host.netloc.replace(':8008', ':8081', 1) + parsed_host.path
    elif parsed_host.port == 8080:
        api_host = parsed_host.scheme + "://" + parsed_host.netloc.replace(':8080', ':8008', 1) + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        files_host = parsed_host.scheme + "://" + parsed_host.netloc.replace(':8080', ':8081', 1) + parsed_host.path
    elif parsed_host.netloc.startswith('demoapp.'):
        # this is our demo server
        api_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('demoapp.', 'demoapi.', 1) + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        files_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('demoapp.', 'demofiles.', 1) + parsed_host.path
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
        files_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('demoapi.', 'demofiles.', 1) + parsed_host.path
    elif parsed_host.netloc.startswith('api.'):
        print('{} is the api server, we need the web server. Replacing \'api.\' with \'app.\''.format(
            parsed_host.netloc))
        api_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('api.', 'app.', 1) + parsed_host.path
        files_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('api.', 'files.', 1) + parsed_host.path
    else:
        api_host = ''
        web_host = ''
        files_host = ''
        if not parsed_host.port:
            print('Host port not detected, do you wish to use the default 8008 port n/[y]? ', end='')
            replace_port = input().lower()
            if not replace_port or replace_port == 'y' or replace_port == 'yes':
                api_host = parsed_host.scheme + "://" + parsed_host.netloc + ':8008' + parsed_host.path
                web_host = parsed_host.scheme + "://" + parsed_host.netloc + ':8080' + parsed_host.path
                files_host = parsed_host.scheme + "://" + parsed_host.netloc + ':8081' + parsed_host.path
        if not api_host:
            api_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path

    api_host = input_url('API Host', api_host)
    files_host = input_url('File Store Host', files_host)

    print('\nTRAINS Hosts configuration:\nAPI: {}\nWeb App: {}\nFile Store: {}\n'.format(
        api_host, web_host, files_host))

    print(description.format(web_host), end='')
    parse_input = input()
    # check if these are valid credentials
    credentials = None
    # noinspection PyBroadException
    try:
        parsed = ConfigFactory.parse_string(parse_input)
        if parsed:
            credentials = parsed.get("credentials", None)
    except Exception:
        credentials = None

    if not credentials or set(credentials) != {"access_key", "secret_key"}:
        print('Could not parse user credentials, try again one after the other.')
        credentials = {}
        # parse individual
        print('Enter user access key: ', end='')
        credentials['access_key'] = input()
        print('Enter user secret: ', end='')
        credentials['secret_key'] = input()

    print('Detected credentials key=\"{}\" secret=\"{}\"'.format(credentials['access_key'],
                                                                 credentials['secret_key'], ))
    # noinspection PyBroadException
    try:
        default_sdk_conf = Path(__file__).parent.absolute() / 'sdk.conf'
        with open(str(default_sdk_conf), 'rt') as f:
            default_sdk = f.read()
    except Exception:
        print('Error! Could not read default configuration file')
        return
    # noinspection PyBroadException
    try:
        with open(str(conf_file), 'wt') as f:
            header = '# TRAINS SDK configuration file\n' \
                     'api {\n' \
                     '    # Notice: \'host\' is the api server (default port 8008), not the web server.\n' \
                     '    api_server: %s\n' \
                     '    web_server: %s\n' \
                     '    files_server: %s\n' \
                     '    # Credentials are generated in the webapp, %s/profile\n' \
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
    print('TRAINS setup completed successfully.')


def input_url(host_type, host=None):
    while True:
        print('{} configured to: [{}] '.format(host_type, host), end='')
        parse_input = input()
        if host and (not parse_input or parse_input.lower() == 'yes' or parse_input.lower() == 'y'):
            break
        if parse_input and verify_url(parse_input):
            host = parse_input
            break
    return host


def verify_url(parse_input):
    try:
        if not parse_input.startswith('http://') and not parse_input.startswith('https://'):
            parse_input = 'http://' + parse_input
        parsed_host = urlparse(parse_input)
        if parsed_host.scheme not in ('http', 'https'):
            parsed_host = None
    except Exception:
        parsed_host = None
        print('Could not parse url {}\nEnter your trains-server host: '.format(parse_input), end='')
    return parsed_host


if __name__ == '__main__':
    main()
