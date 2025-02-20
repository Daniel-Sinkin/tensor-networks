"""
This runner executes all pytenet experiments and stores the resulting images in a specified folder.
"""

import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

from pytenet_experiments.evolution import main as evolution_main
from pytenet_experiments.example import main as example_main
from pytenet_experiments.metts_ising import main as metts_ising_main
from pytenet_experiments.tangent_space import main as tangent_space_main

warnings.filterwarnings("ignore", category=RuntimeWarning)

experiment_main_func_map = {
    "evolution_main": evolution_main,
    "example_main": example_main,
    "metts_ising_main": metts_ising_main,
    "tangent_space_main": tangent_space_main,
}

experiments_folderpath = Path("pytenet_experiments")
experiments_images_folderpath = experiments_folderpath.joinpath("generated")


class PrefixWriter:
    """
    Prefixes all the print statements of the main functions with a \t to seperate runner output
    from the print statement of the main functions.
    """

    def __init__(self, stream, prefix="\t"):
        self.stream = stream
        self.prefix = prefix
        self.at_line_start = True

    def write(self, text):
        """Prefixes the string."""
        for line in text.splitlines(keepends=True):
            if self.at_line_start and line:
                self.stream.write(self.prefix)
            self.stream.write(line)
            self.at_line_start = line.endswith("\n")

    def flush(self):
        """Just use the normal stream flush."""
        self.stream.flush()


def main(image_folderpath: Path = experiments_images_folderpath) -> None:
    """
    Executes all the main functions, using the PrefixWriter to capture the print statements.
    """
    for main_func_name, main_func in experiment_main_func_map.items():
        print(f"Running {main_func_name}")
        with redirect_stdout(PrefixWriter(sys.stdout)):
            main_func(image_folderpath)


if __name__ == "__main__":
    main()
