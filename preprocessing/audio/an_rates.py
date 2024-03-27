import os
import numpy as np
from audio_utils import WaveProcessor
from tqdm import tqdm


def create(
    feature_dir: str,
    tg_folder: str,
    out_folder: str,
    segments_list: list,
    config: dict,
    is_inverted: bool = False
) -> None:
    """ Creates a numpy file with nerve rates.

    This function creates a feature from audio snips. The feature is a 3D array with dimensions
    (n_epochs, 1, n_times). The feature is saved as a .npy file. The filtering pipeline is as follows:
    1. Downsample to 4096 Hz with anti-aliasing filter
    2. Cut off the first and last second of nerve rates
    3. Pad the nerve rates with NaNs to match the length of EEG epochs

    """
    final_epoch_length = config['an_rates']['final_length']
    sfreq_target = config['an_rates']['sfreq']

    feature = np.full((len(segments_list), 1, final_epoch_length * sfreq_target + 1), np.nan)

    desc = 'an rates inverted' if is_inverted else 'an rates'

    for idx, snip_id in enumerate(tqdm(segments_list, desc=desc)):
        processor = WaveProcessor(snip_id, feature_dir, tg_folder, is_subcortex=True)
        processor.downsample(sfreq_goal=sfreq_target, anti_aliasing='1/3')

        sfreq_target, an_rates = processor.get_wave()

        an_rates = an_rates[sfreq_target:-sfreq_target]

        an_rates = WaveProcessor.padding(an_rates, final_epoch_length, sfreq_target, pad_value=np.nan)

        feature[idx, 0, :] = an_rates

    npyfile = 'an_rates_inverted.npy' if is_inverted else 'an_rates.npy'

    np.save(os.path.join(out_folder, npyfile), feature)
