import argparse
from os import path

DEFAULT_EXP = './current_experiment.py'

class ArgParser:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "exp_py_path", type=str, nargs='?', default=DEFAULT_EXP, 
            help="the python script that defines the experiment", 
        )
        args = parser.parse_args()

        self.exp_py_path = args.exp_py_path
        assert path.isfile(self.exp_py_path)
