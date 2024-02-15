import envelopes
import an_rates
from pathlib import Path
import yaml
import mne
mne.set_log_level('WARNING')


def load_config(config_path):
    """ Load the configuration file. """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == '__main__':
    """" Run pipelines to extract envelope and speech features from the audiobook segments.

    Envelopes are directly extracted from the audio files, while speech features are extracted from the output of
    the AN model. AN rates are created twice, they were once extracted from the original speech wave, and once from the
    inverted speech wave. This step of the pipeline is documented in the subfolder `an_model`.

    """
    config = load_config('config.yaml')

    print('Running: ', __file__)

    audiobook_segments = config['audiobook_segments']
    folders = {key: Path(value) for key, value in config['directories'].items()}

    wav_folder = folders['wav_folder']
    an_folder = folders['an_folder']
    an_folder_inverted = folders['an_folder_inverted']
    tg_folder = folders['tg_folder']
    out_folder = folders['out_folder']

    out_folder.mkdir(parents=True, exist_ok=True)

    envelopes.create(wav_folder, tg_folder, out_folder, audiobook_segments, config)
    an_rates.create(an_folder, tg_folder, out_folder, audiobook_segments, config)
    an_rates.create(an_folder_inverted, tg_folder, out_folder, audiobook_segments, config, is_inverted=True)
