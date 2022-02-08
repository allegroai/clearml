import click
from clearml import Task


@click.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name',
              help='The person to greet.')
def hello(count, name):
    task = Task.init(project_name='examples', task_name='Click single command')

    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(count):
        click.echo("Hello {}!".format(name))


if __name__ == '__main__':
    hello()
