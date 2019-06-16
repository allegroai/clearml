from __future__ import print_function

from six.moves import input
from pyhocon import ConfigFactory
from pathlib2 import Path
from six.moves.urllib.parse import urlparse

from trains.backend_api.session.defs import ENV_HOST
from trains.backend_config.defs import LOCAL_CONFIG_FILES
from trains.config import config_obj


description = """
Please create new credentials using the web app: {}/admin
In the Admin page, press "Create new credentials", then press "Copy to clipboard"

Paste credentials here: """

try:
    def_host = ENV_HOST.get(default=config_obj.get("api.host"))
except Exception:
    def_host = 'http://localhost:8080'

host_description = """
Editing configuration file: {CONFIG_FILE}
Enter the url of the trains-server's api service, example: http://localhost:8008 or default demo server [{HOST}]: """.format(
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

    print(host_description, end='')
    parsed_host = None
    while not parsed_host:
        parse_input = input()
        if not parse_input:
            parse_input = def_host
        # noinspection PyBroadException
        try:
            if not parse_input.startswith('http://') and not parse_input.startswith('https://'):
                parse_input = 'http://'+parse_input
            parsed_host = urlparse(parse_input)
            if parsed_host.scheme not in ('http', 'https'):
                parsed_host = None
        except Exception:
            parsed_host = None
            print('Could not parse url {}\nEnter your trains-server host: '.format(parse_input), end='')

    if parsed_host.port == 8080:
        # this is a docker 8080 is the web address, we need the api address, it is 8008
        print('Port 8080 is the web port, we need the api port. Replacing 8080 with 8008')
        api_host = parsed_host.scheme + "://" + parsed_host.netloc.replace(':8080', ':8008') + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
    elif parsed_host.netloc.startswith('demoapp.'):
        print('{} is the web server, we need the api server. Replacing \'demoapp.\' with \'demoapi.\''.format(
            parsed_host.netloc))
        # this is our demo server
        api_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('demoapp.', 'demoapi.') + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
    elif parsed_host.netloc.startswith('app.'):
        print('{} is the web server, we need the api server. Replacing \'app.\' with \'api.\''.format(
            parsed_host.netloc))
        # this is our application server
        api_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('app.', 'api.') + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
    elif parsed_host.port == 8008:
        api_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc.replace(':8008', ':8080') + parsed_host.path
    elif parsed_host.netloc.startswith('demoapi.'):
        api_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('demoapi.', 'demoapp.') + parsed_host.path
    elif parsed_host.netloc.startswith('api.'):
        api_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('api.', 'app.') + parsed_host.path
    else:
        api_host = None
        web_host = None
        if not parsed_host.port:
            print('Host port not detected, do you wish to use the default 8008 port n/[y]? ', end='')
            replace_port = input().lower()
            if not replace_port or replace_port == 'y' or replace_port == 'yes':
                api_host = parsed_host.scheme + "://" + parsed_host.netloc + ':8008' + parsed_host.path
                web_host = parsed_host.scheme + "://" + parsed_host.netloc + ':8080' + parsed_host.path
        if not api_host:
            api_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        if not web_host:
            web_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path

    print('Host configured to: {}'.format(api_host))

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
                     '    host: %s\n' \
                     '    # Credentials are generated in the webapp, %s/admin\n' \
                     '    credentials {"access_key": "%s", "secret_key": "%s"}\n' \
                     '}\n' \
                     'sdk ' % (api_host, web_host, credentials['access_key'], credentials['secret_key'])
            f.write(header)
            f.write(default_sdk)
    except Exception:
        print('Error! Could not write configuration file at: {}'.format(str(conf_file)))
        return

    print('\nNew configuration stored in {}'.format(str(conf_file)))
    print('TRAINS setup completed successfully.')


if __name__ == '__main__':
    main()
