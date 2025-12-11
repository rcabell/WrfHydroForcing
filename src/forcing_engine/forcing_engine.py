import argparse
import sys

from forcing_engine.exceptions import FEException
from forcing_engine.configuration import ConfigOptions

class ForcingEngine:
    def __init__(self, config: ConfigOptions):
        self.config = config

    def generate_forcings(self):
        # Placeholder for the main logic to generate forcings
        self.config.print_config()


# TODO: split into a separate command line tool interface

def gen_forcings_from_command_line():
    parser = argparse.ArgumentParser(
        prog="WrfHydroForcing",
        description="WRF-Hydro Forcing Engine converts meterological input into LDASIN files compatible with WRF-Hydro"
    )

    parser.add_argument('config_file')
    parser.add_argument('-V', '--version')
    parser.add_argument('-C', '--configuration')

    args = parser.parse_args()

    # create configuration object from file
    try:
        config = ConfigOptions(args.config_file)
        config.config_name = args.configuration
        config.version = args.version

        engine = ForcingEngine(config)
        engine.generate_forcings()

    except FEException as fe:
        print(f"Forcing Engine Error ({fe.__class__.__name__}): {fe}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    gen_forcings_from_command_line()