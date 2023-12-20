""" Prepares input for Zilany et al. (2014) model.

This is my pipeline to prepare the speech waves for the Zilany model. The speech waves are normalized, resampled to 100
kHz (as required by the model), and scaled to a given presentation level. The arrays are then saved as .npy files.

"""
import numpy as np
import scipy.signal as signal
from pathlib import Path
from tqdm import tqdm
from scipy.io import wavfile


if __name__ == '__main__':
    from utils import audiobook_segments, parser

    args = parser.parse_args()

    # Path to my `audiofiles` folder where wav files are stored
    SSD_dir = Path('/Volumes/NeuroSSD/Midcortex/data/audiofiles/')
    wav_folder = SSD_dir / 'raw'

    # Path to outpt folders configuration
    if args.input_type == 'inverted':
        folder_out = SSD_dir / 'Zilany_2014/input_arrays_inverted/'
    else:
        folder_out = SSD_dir / 'Zilany_2014/input_arrays/'

    folder_out.mkdir(exist_ok=True)

    # Signal parameters
    SFREQ_COCHLEA = 100e3  # rate for model
    REF_LEVEL = 20e-6  # reference pressure level (20 uPa for humans at 0 dB SPL)
    PRES_LEVEL = 70  # presentation level (70 dB SPL)

    for snip_id in tqdm(audiobook_segments, desc='segments'):
        sfreq, speech_wave = wavfile.read(wav_folder / f'{snip_id}.wav')

        speech_wave = speech_wave.astype(np.float64)
        speech_wave = speech_wave / np.max(np.abs(speech_wave))

        if args.input_type == 'inverted':
            speech_wave = -speech_wave

        # Calculate RMS of normalized speech wave
        speech_rms = np.sqrt(np.mean(speech_wave**2))

        speech_wave = signal.resample(speech_wave, int(len(speech_wave) * (SFREQ_COCHLEA / sfreq)))

        pressure_wave = ((REF_LEVEL / speech_rms) * 10**(PRES_LEVEL / 20.)) * speech_wave

        filename = folder_out / f'{snip_id}_100k_dB.npy'

        np.save(filename, pressure_wave)
