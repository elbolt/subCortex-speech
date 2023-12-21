import os
import numpy as np
from pathlib import Path
from eeg_utils import EEGLoader, EEGDownSegmenter
from utils import default_subjects, parse_arguments, ica, IC_dict

import mne
mne.set_log_level('WARNING')


def run_cortex(raw_dir, out_dir, AEP_out_dir, file_extension, subjects_list):
    """ Applies the prepreprocessing routine to extract the data for cortex encoding analysis and Auditory Evoked
    Potentials (AEP).

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

    AEP pipeline:
        - Copy epochs after final band-pass filter
        - Crop copy to AEP time window and apply baseline correction
        - Average across audiobook segments
        - Save AEP waveform

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

    """
    # Neurophysiology parameters
    TMIN, TMAX = -4.0, 54.0
    FINAL_LENGTH = 48
    SFREQ_GOAL = 128.
    AEP_BASELINE = (-0.200, -0.050)

    # Loop over subjects
    subjects = parse_arguments(subjects_list)

    for subject_id in subjects:
        eeg_loader = EEGLoader(subject_id, raw_dir, file_extension)
        raw = eeg_loader.get_raw()

        # Anti-alias-filter, segment and downsample to 512 Hz
        segmenter = EEGDownSegmenter(
            raw,
            subject_id,
            tmin=TMIN,
            tmax=TMAX,
            decimator=32
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

        del epochs_ica_copy

        epochs.interpolate_bads(reset_bads=True)

        # Anti-alias filter and downsample to 128 Hz
        epochs.filter(
            l_freq=None,
            h_freq=SFREQ_GOAL / 3.0,
            h_trans_bandwidth=SFREQ_GOAL / 10.0,
            method='fir',
            fir_window='hamming',
            phase='zero'
        )
        epochs.decimate(4)

        # 1-9 Hz band-pass filter
        epochs.filter(
            l_freq=1.0,
            h_freq=9.0,
            method='fir',
            fir_window='hamming',
            phase='zero'
        )

        # AEP
        evoked = epochs.copy().crop(tmin=-300e-3, tmax=600e-3)
        evoked.apply_baseline(baseline=(AEP_BASELINE))
        evoked = evoked.average()
        np.save(os.path.join(AEP_out_dir, f'{subject_id}.npy'), evoked.get_data(picks='eeg'))

        # Cut to final length
        epochs.crop(tmin=1.0, tmax=FINAL_LENGTH + 1)

        # Save cortex data
        data = epochs.get_data(picks='eeg')
        filename = os.path.join(out_dir, f'{subject_id}_cortex.npy')
        print(f'Saving: {filename}')
        np.save(filename, data)


if __name__ == '__main__':
    print(f'Running: {__file__}')

    # Path to my `EEG` folder
    SSD_dir = Path('/Volumes/NeuroSSD/subCortex-speech/data/EEG')

    # Path to my `EEG/raw` folder and file extension
    raw_dir = SSD_dir / 'raw'
    file_extension = '_audiobook_raw.fif'

    # Path to out folder where processed data will be stored
    out_dir = SSD_dir / 'TRF/preprocessed/cortex'
    AEP_out_dir = SSD_dir / 'evoked'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(AEP_out_dir, exist_ok=True)

    run_cortex(raw_dir, out_dir, AEP_out_dir, file_extension, default_subjects)
