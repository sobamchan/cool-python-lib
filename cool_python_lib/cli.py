import click

from cool_python_lib.pred import pred
from cool_python_lib.train import train


@click.group()
def cli():
    # This is a group of commands, in following lines we register our commands.
    pass


cli.add_command(train)
cli.add_command(pred)


# This function is the "starting function" exposed to users.
def main():
    cli()
