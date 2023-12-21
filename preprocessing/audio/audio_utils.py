import numpy as np
import mne
import textgrid
from pathlib import Path
from scipy.io import wavfile


class SpeechWaveExtractor():
    """ Class to load audio data from a wav and TextGrid.

    This class is used to load audio data from a wav and TextGrid. The class is initialized with a wav name,
    directory, and TextGrid directory. The wav name is used to load the wav, and the directory and TextGrid
    directory are used to load the wav and TextGrid, respectively.

    Parameters
    ----------
    filename : str
        Name of the wav
    directory : str
        directory containing the wav
    TG_directory : str
        directory containing the Praat TextGrids

    Returns
    -------
    sfreq : float
        Sampling frequency of the wav
    wave : np.ndarray
        Speech wave array of the wav

    """
    def __init__(self, filename, directory, TG_directory):
        self.filename = filename
        self.directory = directory
        self.TG_directory = TG_directory

        self.extract_wave()

    def extract_wave(self):
        """ Gets speech wave from an audio segment and removes silent seconds from the beginning of the segment. """
        filelocation = Path(self.directory) / f'{self.filename}.wav'
        self.sfreq, wave = wavfile.read(filelocation)

        grid_file = Path(self.TG_directory) / f'{self.filename}.TextGrid'
        text_grid = textgrid.TextGrid.fromFile(grid_file)

        silent_seconds = text_grid[0][0].time
        wave = wave[int(silent_seconds * self.sfreq):]

        self.wave = wave.astype(np.float64)

    def get_wave(self):
        return self.sfreq, self.wave


class NerveRatesExtractor(SpeechWaveExtractor):
    """ Subclass of SpeechWaveExtractor to load nerve rates.

    This class is used to load nerve rates from a numpy and TextGrid file. The class is initialized with a filename,
    directory, and TextGrid directory. The filename is used to load the numpy file, and the directory and TextGrid
    directory are used to load the numpy and TextGrid, respectively.

    Parameters
    ----------
    filename : str
        Name of the numpy file
    directory : str
        directory containing the numpy file
    TG_directory : str
        directory containing the Praat TextGrids

    Returns
    -------
    sfreq : float
        Sampling frequency of the nerve rates
    wave : np.ndarray
        Nerve rates array of the numpy file

    """
    def extract_wave(self):
        filelocation = Path(self.directory) / f'{self.filename}_an_rates_44.1k.npy'
        self.sfreq = 44.1e3
        wave = np.load(filelocation)

        grid_file = Path(self.TG_directory) / f'{self.filename}.TextGrid'
        text_grid = textgrid.TextGrid.fromFile(grid_file)

        silent_seconds = text_grid[0][0].time
        wave = wave[int(silent_seconds * self.sfreq):]

        self.wave = wave.astype(np.float64)


class WaveProcessor():
    """ Class to process speech wave arrays.

    This class is used to process speech wave arrays, such as downsampling, filtering, and extracting the Gammatone
    envelope. The class is initialized with a file name, directory, and TextGrid directory. The sampling frequency
    and speech wave signal are updated after each processing step.

    Parameters
    ----------
    filename : str
        Name of the file
    directory : str
        directory containing the file
    TG_directory : str
        directory containing the Praat TextGrids
    is_subcortex : bool | False
        Whether the data is for subcortical analysis or not

    """
    def __init__(self, filename, directory, TG_directory, is_subcortex=False):
        self.is_subcortex = is_subcortex

        if is_subcortex:
            extractor = NerveRatesExtractor(filename=filename, directory=directory, TG_directory=TG_directory)
        else:
            extractor = SpeechWaveExtractor(filename=filename, directory=directory, TG_directory=TG_directory)

        self.sfreq, self.wave = extractor.get_wave()

    def downsample(self, sfreq_goal, highpass=None, anti_aliasing='1/3'):

        lowpass = sfreq_goal / 3.0 if anti_aliasing == '1/3' else int(anti_aliasing)
        l_trans_bandwidth = sfreq_goal / 10.0
        h_trans_bandwidth = None if highpass is None else highpass / 4.0
        phase = 'minimum' if self.is_subcortex else 'zero'

        # Anti-aliasing filter at 1/3 of the new sfreq, with transition bandwidth of 1/10 of the target sfreq
        wave_filtered = mne.filter.filter_data(
            self.wave,
            sfreq=self.sfreq,
            l_freq=highpass,
            l_trans_bandwidth=h_trans_bandwidth,
            h_freq=lowpass,
            h_trans_bandwidth=l_trans_bandwidth,
            method='fir',
            fir_window='hamming',
            phase=phase
        )

        wave_resampled = mne.filter.resample(wave_filtered, down=self.sfreq / sfreq_goal, npad='auto')

        self.sfreq = sfreq_goal
        self.wave = wave_resampled

    def extract_Gammatone_envelope(self, num_filters=24, freq_range=(100, 4000)):
        """ Extracts the Gammatone envelope from the speech wave.

        """
        from scipy import signal

        filterbank = WaveProcessor.gammatone_filterbank(
            sfreq=self.sfreq,
            num_filters=num_filters,
            freq_range=freq_range
        )

        gt_env = np.vstack([signal.filtfilt(filterbank[filt, :], 1.0, self.wave) for filt in range(num_filters)])

        compression = 0.3
        gt_env = np.abs(gt_env)
        gt_env = np.power(gt_env, compression)
        gt_env = np.mean(gt_env, axis=0)

        self.wave = gt_env

    def get_wave(self):
        return self.sfreq, self.wave

    @staticmethod
    def gammatone_filterbank(sfreq, num_filters, freq_range):
        """
        Generate a Gammatone filterbank (Glasberg & Moore, 1990).

        This function generates a Gammatone filterbank, which is a set of bandpass filters that simulate the frequency
        response of the human auditory system. The filters are designed to be similar to the response of the cochlea,
        which is the organ in the inner ear responsible for processing sound.

        Parameters
        ----------
        sfreq : float
            Sampling rate of signal to be filtered.
        num_filters : int
            The number of filters in the filterbank.
        freq_range : tuple of (min_freq, max_freq)
            Frequency range of the filter.

        Returns
        -------
        tuple of (filter_bank, center_freqs)
            A tuple of (filter_bank, center_freqs), where filter_bank is a matrix of shape (num_filters, n), and
            center_freqs is a vector of shape (num_filters,).

        References
        ----------
        Glasberg, B. R., & Moore, B. C. (1990). Derivation of auditory filter shapes from notched-noise data.
        Hearing Research, 47(1-2), 103-138. doi:10.1016/0378-5955(90)90170-T

        """

        # Compute ERB (Equivalent Rectangular Bandwidth)
        min_freq, max_freq = freq_range
        erb_min = 24.7 * (4.37 * min_freq / 1000 + 1)
        erb_max = 24.7 * (4.37 * max_freq / 1000 + 1)

        # Compute center frequencies in ERB and Hz
        center_freqs_erb = np.linspace(erb_min, erb_max, num_filters)
        center_freqs_hz = (center_freqs_erb / 24.7 - 1) / 4.37 * 1000

        # Compute filter bandwidths and Q factors
        q = 1.0 / (center_freqs_erb * 0.00437 + 1.0)
        bandwidths = center_freqs_hz * q

        # Compute filter bank
        filter_bank = np.zeros((num_filters, 4))
        t = np.arange(4) / sfreq
        for i in range(num_filters):
            c = 2 * np.pi * center_freqs_hz[i]
            b = 1.019 * 2 * np.pi * bandwidths[i]

            # Compute envelope and sine wave
            envelope = (c ** 4) / (b * np.math.factorial(4)) * t ** 3 * np.exp(-b * t)
            sine_wave = np.sin(c * t)

            # Apply envelope to sine wave and store in filter bank
            filter_bank[i, :] = sine_wave * envelope

        return filter_bank

    @staticmethod
    def padding(array, length, sfreq, pad_value=np.nan):
        """ Pads the input array with either NaN or 0 values to match the desired length in seconds,
        based on the given sampling frequency.

        Parameters
        ----------
        array : np.ndarray
            Array to be padded.
        length : float
            Desired length of the array in seconds.
        sfreq : float
            Sampling frequency of the array.

        Returns
        -------
        array_padded : np.ndarray
            Padded array.

        """
        padding_length = length * sfreq - array.shape[0] + 1  # + 1 to account for difference with EEG
        array_padded = np.pad(array, (0, padding_length), constant_values=pad_value)

        return array_padded
