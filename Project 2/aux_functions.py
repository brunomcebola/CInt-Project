import os
import yaml

import pandas as pd


def read_yaml(file_path: str):
    try:
        with open(file_path, "r") as stream:
            try:
                configs = yaml.safe_load(stream)

                if configs == None:
                    raise SystemExit("The configuration file '%s' is empty." % file_path)

                return configs

            except:
                raise SystemExit("Error reading configuration file '%s'." % file_path)
    except:
        raise SystemExit("Configuration file '%s' not found'." % file_path)


def validate_config(config: dict, keys: list, file_path: str):
    for key in keys:
        if key not in config or not config[key]:
            raise SystemExit("Parameter '%s' missing in configuration file '%s'." % (key, file_path))


def readCSV(file_path):
    if not os.path.exists(file_path):
        raise SystemExit("File '%s' does not exist" % file_path)

    return pd.read_csv(file_path).iloc[:, 1:].values.tolist()
