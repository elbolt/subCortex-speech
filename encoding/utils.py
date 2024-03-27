import yaml


def load_config(config_path: str) -> dict:
    """ Load configuration from YAML file. """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def parse_arguments(default_subjects: list) -> list:
    """ Argument parser for parsing from command-line.

    """
    import argparse

    parser = argparse.ArgumentParser(description='Parsing subjects for encoding analysis.')
    parser.add_argument('-s', '--subjects', type=str, help='Subject numbers separated by commas (no spaces)')
    args = parser.parse_args()

    if args.subjects:
        subjects = args.subjects.split(',')
        subjects = [subject.strip() for subject in subjects]
    else:
        subjects = default_subjects

    return subjects
