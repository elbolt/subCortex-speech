import os
import numpy as np
from pathlib import Path
from eeg_utils import EEGLoader, EEGDownSegmenter
from utils import default_subjects, parse_arguments, load_config

import mne
mne.set_log_level('WARNING')


def abr_pipeline(raw_dir, out_dir, file_extension, default_subjects, config):
    """ Applies the prepreprocessing routine to extract the Auditory Brainstem Response (ABR) response. The filtering
    pipeline is identical to `run_cortex.py`.

    Pipeline:
        - Load raw data (`EEGLoader` takes care of channel configuration)
        - Notch filter to remove line noise
        - Segment raw EEG into epochs and downsample to 4096 Hz with anti-aliasing filtering using `EEGDownSegmenter`
        - High-pass filter at 80 Hz to remove cortical contributions
        - Apply baseline correction
        - Reject epochs exceeding 40 µV at the vertex channel
        - Average epochs
        - Save ABR response

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
    # ABR epochs parameters
    abr_min = config['abr']['min']
    abr_max = config['abr']['max']
    abr_baseline = (
        config['abr']['baseline']['start'],
        config['abr']['baseline']['end']
    )

    # Neurophysiology parameters for filters and rates
    notch_frequencies = np.array(config['neurophysiology']['notch']['frequencies'])
    notch_width = config['neurophysiology']['notch']['width']
    subcortex_highpass = config['neurophysiology']['subcortex']['highpass']
    subcortex_sfreq = config['neurophysiology']['subcortex']['sfreq']
    clean_threshold_abr = config['neurophysiology']['clean_threshold']['abr']

    subjects = parse_arguments(default_subjects)
    # no_epochs = np.zeros(len(subjects)) * np.nan

    # Loop over subjects
    for _, subject_id in enumerate(subjects):
        eeg_loader = EEGLoader(subject_id, raw_dir, file_extension, is_subcortex=True, is_ABR=True)
        raw, vertex_channel = eeg_loader.get_raw()

        # Notch filter to remove line noise
        for _, freq in enumerate(notch_frequencies):
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
            tmin=abr_min,
            tmax=abr_max,
            decimator=decimator,
            highpass=subcortex_highpass,
            is_subcortex=True,
            is_ABR=True
        )

        epochs = segmenter.get_epochs()

        # Baseline correction
        epochs.apply_baseline(abr_baseline)

        # Pick vertex channel only
        epochs = epochs.copy().pick_channels([vertex_channel])

        # Reject epochs exceeding 40 uV at the vertex channel
        epochs.drop_bad(reject=dict(eeg=clean_threshold_abr))

        # print(f'No. of epochs: {epochs.get_data(copy=True).shape[0]}')
        # no_epochs[_] = epochs.get_data(copy=True).shape[0]

        # Average epochs
        evoked = epochs.copy().average()

        # Save ABR response
        data = evoked.get_data()
        filename = os.path.join(out_dir, f'{subject_id}.npy')
        print(f'Saving: {filename}')
        np.save(filename, data)
        # np.save('no_epochs.npy', no_epochs)


if __name__ == '__main__':
    print(f'Running: {__file__}')

    config = load_config('config.yaml')
    eeg_config = load_config('eeg_config.yaml')

    # Paths to folders
    folders = {key: Path(value) for key, value in config['directories'].items()}
    raw_dir = folders['abr_raw_dir']
    out_dir = folders['abr_dir']
    os.makedirs(out_dir, exist_ok=True)

    file_extension = config['file_extensions']['abr']

    abr_pipeline(raw_dir, out_dir, file_extension, default_subjects, eeg_config)
