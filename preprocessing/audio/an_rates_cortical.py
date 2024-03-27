import os
import mne
import numpy as np
from audio_utils import WaveProcessor
from tqdm import tqdm


def create(wav_folder: str, tg_folder: str, out_folder: str, segments_list: list, config: dict) -> None:
    """ Create feature from audio snips.

    This function creates a feature from audio snips. The feature is a 3D array with dimensions
    (n_epochs, 1, n_times). The feature is saved as a .npy file. The filtering pipeline is as follows:
    1. Downsample to 12000 Hz with anti-aliasing filter
    2. Extract Gammatone envelope
    2. Downsample to 512 Hz
    3. Finally, downsample to target sfreq of 64 Hz
    4. Band-pass filter envelope at 1-9 Hz
    5. Cut off the first and last second of envelope
    7. Pad the envelope with NaNs to match the length of EEG epochs

    Parameters
    ----------
    segments_list : list
        List of audio snips
    wav_folder : str
        Folder containing the wav files
    tg_folder : str
        Folder containing the Praat TextGrids
    out_folder : str
        Folder to save the feature
    config : dict
        Configuration dictionary

    """
    final_epoch_length = config['envelope']['final_length']
    sfreq_target = config['envelope']['sfreq']
    final_bandpass = (
        config['envelope']['bandpass_start'],
        config['envelope']['bandpass_stop']
    )

    feature = np.full((len(segments_list), 1, final_epoch_length * sfreq_target + 1), np.nan)

    for idx, snip_id in enumerate(tqdm(segments_list, desc='cortical an rates')):
        processor = WaveProcessor(snip_id, wav_folder, tg_folder, use_nerve_rates=True)
        processor.downsample(sfreq_goal=512)
        processor.downsample(sfreq_goal=sfreq_target)

        sfreq_target, an_rates = processor.get_wave()

        an_rates = mne.filter.filter_data(
            an_rates,
            sfreq=sfreq_target,
            l_freq=final_bandpass[0],
            h_freq=final_bandpass[1],
            method='fir',
            fir_window='hamming',
            phase='zero'
        )

        an_rates = an_rates[sfreq_target:-sfreq_target]

        an_rates = WaveProcessor.padding(an_rates, final_epoch_length, sfreq_target, pad_value=np.nan)

        feature[idx, 0, :] = an_rates

    np.save(os.path.join(out_folder, 'an_rates_cortical'), feature)
