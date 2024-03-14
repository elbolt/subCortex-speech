import os
import numpy as np
from pathlib import Path
from eeg_utils import EEGLoader, EEGDownSegmenter, clean_subcortex_signal
from utils import parse_arguments, load_config

import mne
mne.set_log_level('WARNING')


def subcortex_pipeline(raw_dir, out_dir, file_extension, subjects_list, config):
    """ Applies the prepreprocessing routine to extract the data for subcortex encoding analysis.

    Pipeline:
        - Load raw data (`EEGLoader` takes care of channel configuration)
        - Notch filter to remove line noise
        - Segment raw EEG into epochs and downsample to 4096 Hz with anti-aliasing filtering using `EEGDownSegmenter`
        - High-pass filter at 80 Hz to remove cortical contributions
        - Cut to final length of 48 s
        - Get data and clean out segments exceeding 100 µV
        - Save subcortex data for encoding analysis

    Parameters
    ----------
    raw_dir : str
        Path to raw EEG data.
    file_extension : str
        File extension of raw EEG data files.
    out_dir : str
        Path to out folder where preprocessed data will be stored.
    subjects_list : list
        List of participant IDs to be processed.
    config : dict
        Configuration dictionary.

    """
    # Speech epochs parameters
    speech_epochs_min = config['speech_epochs']['min']
    speech_epochs_max = config['speech_epochs']['max']
    final_epoch_length = config['speech_epochs']['final_epoch_length']
    subjects = config['subjects']

    # Neuropysiology parameters for filters and rates
    subcortex_highpass = config['neurophysiology']['subcortex']['highpass']
    subcortex_sfreq = config['neurophysiology']['subcortex']['sfreq']
    notch_frequencies = np.array(config['neurophysiology']['notch']['frequencies'])
    notch_width = config['neurophysiology']['notch']['width']
    clean_threshold = config['neurophysiology']['clean_threshold']['subcortex']

    subjects = parse_arguments(subjects_list)

    # Loop over subjects
    for subject_id in subjects:
        eeg_loader = EEGLoader(subject_id, raw_dir, file_extension, is_subcortex=True)
        raw, vertex_channel = eeg_loader.get_raw()

        # Notch filter to remove line noise
        for f, freq in enumerate(notch_frequencies):
            raw.notch_filter(
                freqs=freq,
                method='iir',
                iir_params=dict(order=2, ftype='butter', output='sos'),
                notch_widths=notch_width
            )

        # Anti-alias-filter, segment and downsample to 4096 Hz
        decimator = int(raw.info['sfreq'] / subcortex_sfreq)

        segmenter = EEGDownSegmenter(
            raw,
            subject_id,
            tmin=speech_epochs_min,
            tmax=speech_epochs_max,
            decimator=decimator,
            highpass=subcortex_highpass,
            is_subcortex=True,
            is_ABR=False,
        )
        epochs = segmenter.get_epochs()

        # Cut to final length
        epochs.crop(tmin=1.0, tmax=final_epoch_length + 1)

        # Clean out segments exceeding 100 µV
        data = epochs.get_data(picks=vertex_channel)
        data_cleaned = clean_subcortex_signal(data, subcortex_sfreq, threshold=clean_threshold, segment_duration=1.0)

        # Save subcortex data
        filename = os.path.join(out_dir, f'{subject_id}_subcortex.npy')
        print(f'Saving: {filename}')
        np.save(filename, data_cleaned)


if __name__ == '__main__':
    print(f'Running: {__file__}')

    config = load_config('config.yaml')
    eeg_config = load_config('eeg_config.yaml')
    subjects = config['subjects']

    # Paths to folders
    folders = {key: Path(value) for key, value in config['directories'].items()}
    raw_dir = folders['eeg_raw_dir']
    out_dir = folders['eeg_subcortex_dir']
    os.makedirs(out_dir, exist_ok=True)

    file_extension = config['file_extensions']['eeg']

    subcortex_pipeline(raw_dir, out_dir, file_extension, subjects, eeg_config)
