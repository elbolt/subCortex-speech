""" Process output of Zilany et al. (2014) model.

The an rate arrays are read in as .npy files, and the AN firing rates are resampled to 44.1 kHz, averaged across all
AN frequencies, and shifted by 1 ms to produce the final AN rate. The arrays are then saved as .npy files.

"""
import numpy as np
import scipy.signal as signal
import ic_cn2018 as nuclei
from pathlib import Path
from tqdm import tqdm


if __name__ == '__main__':
    from utils import audiobook_segments, parser

    args = parser.parse_args()

    # Path to my `Zilany_2014` folder
    DIRECTORY = Path('/Volumes/NeuroSSD/Midcortex/data/audiofiles/Zilany_2014')

    # Path to outpt folders configuration
    if args.input_type == 'inverted':
        folder_in = DIRECTORY / 'an_rate_arrays_inverted'
        folder_out = DIRECTORY / 'anm_features_inverted'
    else:
        folder_in = DIRECTORY / 'an_rate_arrays'
        folder_out = DIRECTORY / 'anm_features'

    folder_out.mkdir(exist_ok=True)

    # Signal parameters
    SFREQ_COCHLEA = 100e3  # model rate
    SFREQ = 44.1e3  # audio rate
    FINAL_SHIFT = int(SFREQ * 1e-3)  # shift wave by 1 ms

    for snip_id in tqdm(audiobook_segments, desc='segments'):
        an_rates = np.load(folder_in / f'{snip_id}_an_rates_100k.npy')

        initial_wave_length = int(len(an_rates) / SFREQ_COCHLEA * SFREQ)

        an_rates = signal.resample(an_rates, initial_wave_length, axis=0)

        # Shift, scale, and sum auditory nerve firing rates to produce final auditory nerve wave
        an_rates = nuclei.M1 * an_rates.mean(axis=1)
        an_rates = np.roll(an_rates, FINAL_SHIFT)
        an_rates[:FINAL_SHIFT] = an_rates[FINAL_SHIFT + 1]

        filename = folder_out / f'{snip_id}_an_rates_44.1k.npy'

        np.save(filename, an_rates)
