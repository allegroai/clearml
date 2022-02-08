import click
from clearml import Task


@click.group()
def cli():
    task = Task.init(project_name='examples', task_name='Click multi command')
    print('done')


@cli.command('hello', help='test help')
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name', help='The person to greet.')
def hello(count, name):
    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(count):
        click.echo("Hello {}!".format(name))
    print('done')


CONTEXT_SETTINGS = dict(
    default_map={'runserver': {'port': 5000}}
)


@cli.command('runserver')
@click.option('--port', default=8000)
@click.option('--name', help='service name')
def runserver(port, name):
    click.echo("Serving on http://127.0.0.1:{} {}/".format(port, name))


if __name__ == '__main__':
    cli()
