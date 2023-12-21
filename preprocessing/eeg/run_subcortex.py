import os
import numpy as np
from pathlib import Path
from eeg_utils import EEGLoader, EEGDownSegmenter, clean_subcortex_signal
from utils import default_subjects, parse_arguments

import mne
mne.set_log_level('WARNING')


def run_subcortex(raw_dir, out_dir, file_extension, subjects_list):
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

    """
    # Neurophysiology parameters
    TMIN, TMAX = -4.0, 54.0
    FINAL_LENGTH = 48
    SFREQ_TARGET = 4096.
    NOTCH_FREQUENCIES = np.arange(50., (1000. + 1), 50.)
    NOTCH_WIDTH = 5

    subjects = parse_arguments(subjects_list)

    # Loop over subjects
    for subject_id in subjects:
        eeg_loader = EEGLoader(subject_id, raw_dir, file_extension, is_subcortex=True)
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
            is_ABR=False,
        )
        epochs = segmenter.get_epochs()

        # Cut to final length
        epochs.crop(tmin=1.0, tmax=FINAL_LENGTH + 1)

        # Clean out segments exceeding 100 µV
        data = epochs.get_data(picks=vertex_channel)
        data_cleaned = clean_subcortex_signal(data, SFREQ_TARGET, threshold=100., segment_duration=1.0)

        # Save subcortex data
        filename = os.path.join(out_dir, f'{subject_id}_subcortex.npy')
        print(f'Saving: {filename}')
        np.save(filename, data_cleaned)


if __name__ == '__main__':
    print(f'Running: {__file__}')

    # Path to my `EEG` folder where data is stored in subfolders
    SSD_dir = Path('/Volumes/NeuroSSD/subCortex-speech/data/EEG/')

    # Path to my `EEG/raw` folder and file extension
    raw_dir = SSD_dir / 'raw'
    file_extension = '_audiobook_raw.fif'

    # Path to out folder where processed data will be stored
    out_dir = SSD_dir / 'TRF/preprocessed/subcortex'
    os.makedirs(out_dir, exist_ok=True)

    run_subcortex(raw_dir, out_dir, file_extension, default_subjects)
