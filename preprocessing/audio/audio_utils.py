import numpy as np
import mne
import textgrid
from pathlib import Path
from scipy.io import wavfile


class SpeechWaveExtractor:
    """ Class to load audio data from a wav and TextGrid. """
    def __init__(self, filename, wav_dir, tg_dir):
        """
        Initialize the SpeechWaveExtractor.

        My wavs are stereo wav files, with the first channel containing the speech and the second channel containing
        an impulse vector coding for speech onest. The impulse time point is documented in the TextGrid file since
        I used Praat to annotate the speech onset. Thus, the file `filename.wav` should be a stereo wav file, and the
        file `filename.TextGrid` should contain the annotation of the speech onset in the first interval of the first
        tier.
        I structered the class such that the names of the wav and TextGrid files are the same.

        Parameters
        ----------
        filename : str
            Name of the wav
        wav_dir : str
            Directory containing the wav
        tg_dir : str
            Directory containing the Praat TextGrids
        """
        self.filename = filename
        self.wav_dir = wav_dir
        self.tg_dir = tg_dir
        self.sfreq = None
        self.wave = None

    def extract_wave(self):
        """
        Extract speech wave from an audio segment and remove silent seconds from the beginning of the segment.
        """
        filelocation = Path(self.wav_dir) / f'{self.filename}.wav'
        sfreq, wave = wavfile.read(filelocation)

        if wave.ndim > 1:
            wave = wave[:, 0]  # first channel, if stereo

        grid_file = Path(self.tg_dir) / f'{self.filename}.TextGrid'
        text_grid = textgrid.TextGrid.fromFile(grid_file)

        silent_seconds = text_grid[0][0].time
        wave = wave[int(silent_seconds * sfreq):]

        self.sfreq = sfreq
        self.wave = wave.astype(np.float64)

    def get_wave(self):
        """
        Get the speech wave.

        Returns
        -------
        sfreq : float
            Sampling frequency of the wav
        wave : np.ndarray
            Speech wave array of the wav
        """
        if self.sfreq is None or self.wave is None:
            self.extract_wave()

        return self.sfreq, self.wave


class NerveRatesExtractor(SpeechWaveExtractor):
    """ Subclass of SpeechWaveExtractor to load nerve rates. """
    def extract_wave(self):
        """
        Extract speech wave from an audio segment and remove silent seconds from the beginning of the segment.

        Note that I set `sfreq` manually to 44.1 kHz.
        """
        filelocation = Path(self.wav_dir) / f'{self.filename}_an_rates_44.1k.npy'
        self.sfreq = 44.1e3
        wave = np.load(filelocation)

        grid_file = Path(self.tg_dir) / f'{self.filename}.TextGrid'
        text_grid = textgrid.TextGrid.fromFile(grid_file)

        silent_seconds = text_grid[0][0].time
        wave = wave[int(silent_seconds * self.sfreq):]

        self.wave = wave.astype(np.float64)


class WaveProcessor():
    """ Class to process speech wave arrays.

    This class is used to process speech wave arrays, such as downsampling, filtering, and extracting the Gammatone
    envelope. The class is initialized with a file name, directory, and TextGrid directory. The sampling frequency
    and speech wave signal are updated after each processing step.

    """
    def __init__(self, filename, wav_dir, tg_dir, is_subcortex=False):
        """ Initialize the WaveProcessor.

        Parameters
        ----------
        filename : str
            Name of the wav
        wav_dir : str
            Directory containing the wav
        tg_dir : str
            Directory containing the Praat TextGrids
        is_subcortex : bool
            If True, the speech wave is assumed to be the output of the AN model.
        """
        self.is_subcortex = is_subcortex

        if is_subcortex:
            extractor = NerveRatesExtractor(filename=filename, wav_dir=wav_dir, tg_dir=tg_dir)
        else:
            extractor = SpeechWaveExtractor(filename=filename, wav_dir=wav_dir, tg_dir=tg_dir)

        self.sfreq, self.wave = extractor.get_wave()

    def downsample(self, sfreq_goal, highpass=None, anti_aliasing='1/3'):
        """ Downsample the speech wave to the target sampling frequency.

        Parameters
        ----------
        sfreq_goal : float
            Target sampling frequency.
        highpass : float
            Highpass frequency for anti-aliasing filter.
        anti_aliasing : str
            Anti-aliasing filter cutoff. If '1/3', the cutoff is set to 1/3 of the target sfreq,
            otherwise it is set to the given value.

        """
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
        """ Extracts the Gammatone envelope from the speech wave. """
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
