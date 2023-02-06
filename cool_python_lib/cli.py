import click

from cool_python_lib.train import train
from cool_python_lib.pred import pred


@click.group()
def cli():
    pass


cli.add_command(train)
cli.add_command(pred)


def main():
    cli()
