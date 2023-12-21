import os
import numpy as np
from audio_utils import WaveProcessor
from tqdm import tqdm


def create(FEATURE_DIR, TG_DIR, out_dir, segments_list, is_inverted=False):
    """ Creates a numpy file with nerve rates. """

    # feature parameters
    FINAL_LENGTH = 48
    SFREQ_TARGET = 4096

    feature = np.full((len(segments_list), 1, FINAL_LENGTH * SFREQ_TARGET + 1), np.nan)

    desc = 'an rates inverted' if is_inverted else 'an rates'

    for idx, snip_id in enumerate(tqdm(segments_list, desc=desc)):
        processor = WaveProcessor(snip_id, FEATURE_DIR, TG_DIR, is_subcortex=True)
        processor.downsample(sfreq_goal=SFREQ_TARGET, anti_aliasing='1/3')

        SFREQ_TARGET, an_rates = processor.get_wave()

        # Discard fist and last second
        an_rates = an_rates[SFREQ_TARGET:-SFREQ_TARGET]

        an_rates = WaveProcessor.padding(an_rates, FINAL_LENGTH, SFREQ_TARGET, pad_value=np.nan)

        feature[idx, 0, :] = an_rates

    npyfile = 'an_rates_inverted.npy' if is_inverted else 'an_rates.npy'

    np.save(os.path.join(out_dir, npyfile), feature)
