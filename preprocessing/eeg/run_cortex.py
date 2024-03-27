import os
import numpy as np
from pathlib import Path
from eeg_utils import EEGLoader, EEGDownSegmenter
from utils import parse_arguments, ica, IC_dict, load_config

import mne
mne.set_log_level('WARNING')


def cortex_pipeline(
    raw_dir: str,
    out_dir: str,
    aep_out_dir: str,
    file_extension: str,
    subjects_list: list,
    config: dict
) -> None:
    """ Applies the prepreprocessing routine to extract the data for cortex encoding analysis and Auditory Evoked
    Potentials (aep).

    Pipeline:
        - Load raw data (`EEGLoader` takes care of channel configuration)
        - Segment raw EEG into epochs and downsample to 4096 Hz with anti-aliasing filtering using `EEGDownSegmenter`
        - Run ICA on high-pass filtered epochs copy
        - Zero out ICA componends in original epochs
        - Interpolate bad channels
        - Anti-alias filter and downsample to 128 Hz
        - 1-9 Hz band-pass filter
        - Cut to final length of 48 s
        - Save cortex data for encoding analysis

    aep pipeline:
        - Copy epochs after final band-pass filter
        - Crop copy to aep time window and apply baseline correction
        - Average across audiobook segments
        - Save aep waveform

    Parameters
    ----------
    raw_dir : str
        Path to raw EEG data.
    file_extension : str
        File extension of raw EEG data files.
    out_dir : str
        Path to out folder where preprocessed data will be stored.
    subjects_list : list
        List of participant IDs to be processed.\
    config : dict
        Configuration dictionary.

    """
    # Speech epochs parameters
    speech_epochs_min = config['speech_epochs']['min']
    speech_epochs_max = config['speech_epochs']['max']
    aep_min = config['aep']['min']
    aep_max = config['aep']['max']
    aep_baseline = (
        config['aep']['baseline']['start'],
        config['aep']['baseline']['end']
    )
    final_epoch_length = config['speech_epochs']['final_epoch_length']

    # Neurophysiology parameters for filters and rates
    cortex_bandpass = (
        config['neurophysiology']['cortex']['bandpass']['low'],
        config['neurophysiology']['cortex']['bandpass']['high']
    )
    ica_sfreq = config['neurophysiology']['cortex']['ica_sfreq']
    cortex_sfreq = config['neurophysiology']['cortex']['sfreq']

    # Loop over subjects
    subjects = parse_arguments(subjects_list)

    for subject_id in subjects:
        eeg_loader = EEGLoader(subject_id, raw_dir, file_extension)
        raw = eeg_loader.get_raw()

        # Anti-alias-filter, segment and downsample to 512 Hz
        decimator = int(raw.info['sfreq'] / ica_sfreq)
        segmenter = EEGDownSegmenter(
            raw,
            subject_id,
            tmin=speech_epochs_min,
            tmax=speech_epochs_max,
            decimator=decimator,
        )
        epochs = segmenter.get_epochs()

        # Run ICA on high-pass filtered epochs copy
        epochs_ica_copy = epochs.copy()
        epochs_ica_copy.filter(
            l_freq=1.0,
            h_freq=None,
            method='fir',
            fir_window='hamming',
            phase='zero'
        )
        ica.fit(epochs_ica_copy)
        ica.exclude = IC_dict[subject_id]
        ica.apply(epochs)

        del epochs_ica_copy, decimator

        epochs.interpolate_bads(reset_bads=True)

        # Anti-alias filter and downsample to 128 Hz
        epochs.filter(
            l_freq=None,
            h_freq=cortex_sfreq / 3.0,
            h_trans_bandwidth=cortex_sfreq / 10.0,
            method='fir',
            fir_window='hamming',
            phase='zero'
        )

        decimator = int(ica_sfreq / cortex_sfreq)

        epochs.decimate(decimator)

        # Final Hz band-pass filter
        epochs.filter(
            l_freq=cortex_bandpass[0],
            h_freq=cortex_bandpass[1],
            method='fir',
            fir_window='hamming',
            phase='zero'
        )

        # AEP
        evoked = epochs.copy().crop(tmin=aep_min, tmax=aep_max)
        evoked.apply_baseline(baseline=(aep_baseline))
        evoked = evoked.average()
        np.save(os.path.join(aep_out_dir, f'{subject_id}.npy'), evoked.get_data(picks='eeg'))

        # Cut to final length
        epochs.crop(tmin=1.0, tmax=final_epoch_length + 1)

        # Save cortex data
        data = epochs.get_data(picks='eeg')
        filename = os.path.join(out_dir, f'{subject_id}_cortex.npy')
        print(f'Saving: {filename}')
        np.save(filename, data)


if __name__ == '__main__':
    print(f'Running: {__file__}')

    config = load_config('config.yaml')
    eeg_config = load_config('eeg_config.yaml')
    subjects = config['subjects']

    # Paths to folders
    folders = {key: Path(value) for key, value in config['directories'].items()}
    raw_dir = folders['eeg_raw_dir']
    out_dir = folders['eeg_cortex_dir']
    aep_out_dir = folders['aep_dir']
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(aep_out_dir, exist_ok=True)

    file_extension = config['file_extensions']['eeg']

    cortex_pipeline(raw_dir, out_dir, aep_out_dir, file_extension, subjects, eeg_config)
