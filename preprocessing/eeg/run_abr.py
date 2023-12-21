import os
import numpy as np
from pathlib import Path
from eeg_utils import EEGLoader, EEGDownSegmenter
from utils import default_subjects, parse_arguments

import mne
mne.set_log_level('WARNING')


def run_abr(raw_dir, out_dir, file_extension, default_subjects):
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

    """
    # Neurophysiology parameters
    TMIN, TMAX = -10e-3, 30e-3
    BASELINE = (-10e-3, -5e-3)
    NOTCH_FREQUENCIES = np.arange(50., (1000. + 1), 50.)
    NOTCH_WIDTH = 5

    subjects = parse_arguments(default_subjects)
    no_epochs = np.zeros(len(subjects)) * np.nan

    # Loop over subjects
    for idx, subject_id in enumerate(subjects):
        eeg_loader = EEGLoader(subject_id, raw_dir, file_extension, is_subcortex=True, is_ABR=True)
        raw, vertex_channel = eeg_loader.get_raw()

        # Notch filter to remove line noise
        for f, freq in enumerate(NOTCH_FREQUENCIES):
            raw.notch_filter(
                freqs=freq,
                method='iir',
                iir_params=dict(order=2, ftype='butter', output='sos'),
                notch_widths=NOTCH_WIDTH
            )

        # Anti-alias-filter, segment and downsample to 4096 Hz
        segmenter = EEGDownSegmenter(
            raw,
            subject_id,
            tmin=TMIN,
            tmax=TMAX,
            decimator=4,
            highpass=80,
            is_subcortex=True,
            is_ABR=True
        )

        epochs = segmenter.get_epochs()

        # Baseline correction
        epochs.apply_baseline(BASELINE)

        # Pick vertex channel only
        epochs = epochs.copy().pick_channels([vertex_channel])

        # Reject epochs exceeding 40 uV at the vertex channel
        epochs.drop_bad(reject=dict(eeg=40e-6))

        print(f'No. of epochs: {epochs.get_data(copy=True).shape[0]}')
        no_epochs[idx] = epochs.get_data(copy=True).shape[0]

        # Average epochs
        evoked = epochs.copy().average()

        # Save ABR response
        data = evoked.get_data()
        filename = os.path.join(out_dir, f'{subject_id}.npy')
        print(f'Saving: {filename}')
        np.save(filename, data)
        np.save('no_epochs.npy', no_epochs)


if __name__ == '__main__':
    print(f'Running: {__file__}')

    # Path to my `ABR` folder where data is stored in subfolders
    SSD_dir = Path('/Volumes/NeuroSSD/subCortex-speech/data/ABR/')

    # Path to my `EEG/raw` folder and file extension
    raw_dir = SSD_dir / 'raw'
    file_extension = '_ABR_raw.fif'

    # Path to out folder where processed data will be stored
    out_dir = SSD_dir / 'evoked'
    os.makedirs(out_dir, exist_ok=True)

    run_abr(raw_dir, out_dir, file_extension, default_subjects)
