import os
import numpy as np

import mne


class EEGLoader():
    """ Class to load raw EEG data.

    The EEG data is loaded and bad as well as reference channels are set based on previous visual
    data inspection. The reference channels are set to the average mastoids or to T7 and T8, depending on the subject.
    For subcortical analysis, the raw object will only contain the vertex channel, the Status channel and the reference.
    Vertex channel is set to Cz, or if Cz is bad, to Pz. The data is returned as MNE.raw object with bad channels and
    reference channels already set.

    Parameters
    ----------
    subject_id : str
        Subject ID
    directory : str
        directory containing the fif file
    file_extension : str
        File extension of the fif file
    is_subcortex : bool | False
        Whether the data is for subcortical analysis or not

    Returns
    -------
    raw : mne.io.Raw
        Raw EEG data

    Errors
    ------
    ValueError
        If bad channels and reference channels overlap

    Examples
    --------
    >>> eeg_loader = EEGLoader('p01', 'data', '.fif', is_subcortex=False)
    >>> raw = eeg_loader.get_raw()

    """
    def __init__(self, subject_id, directory, file_extension, is_subcortex=False, is_ABR=False):
        self.subject_id = subject_id
        self.directory = directory
        self.filename = f'{self.subject_id}{file_extension}'
        self.filelocation = os.path.join(self.directory, self.filename)
        self.is_subcortex_ = is_subcortex
        self.is_ABR = is_ABR
        self.raw_ = mne.io.read_raw_fif(self.filelocation, preload=False)

        self.bad_channels = None
        self.ref_channels = None

        if self.is_ABR and not self.is_subcortex_:
            raise ValueError('is_ABR can only be true if is_subcortex is true')

        self.configure_channels()
        self.set_reference()

    def configure_channels(self):
        """ Sets bad channels and reference channels.

        """
        from utils import bad_audio_channels_dict, bad_ABR_channels_dict, subjects_bad_audio_refs, subjects_bad_ABR_refs

        # Bad channels (from visual inspection)
        bad_channels_dict = bad_ABR_channels_dict if self.is_ABR else bad_audio_channels_dict
        subjects_bad_refs = subjects_bad_ABR_refs if self.is_ABR else subjects_bad_audio_refs

        self.bad_channels = bad_channels_dict[self.subject_id]
        self.raw_.info['bads'] = self.bad_channels

        # Reference channels (mastoids or T7 and T8)
        if self.subject_id in subjects_bad_refs:
            self.ref_channels = ['T7', 'T8']
            print(f'Using reference channels: {self.ref_channels}.')
        else:
            self.ref_channels = ['EXG3', 'EXG4']

        if self.subject_id in subjects_bad_refs and set(self.ref_channels).issubset(set(self.bad_channels)):
            print(f'Bad channels and reference channels overlap for subject {self.subject_id}!')

    def set_reference(self):
        """ Loads data into memory and sets reference channels. """
        self.raw_.load_data()

        if self.is_subcortex_:
            vertex_channel = 'Pz' if 'Cz' in self.bad_channels else 'Cz'
            if vertex_channel != 'Cz':
                print(f'Using vertex channel: {vertex_channel}.')

            subcortex_channels = [vertex_channel, 'Status'] + self.ref_channels
            self.raw_ = self.raw_.pick_channels(subcortex_channels, ordered=True)
            self.vertex_channel = vertex_channel

        self.raw_.set_eeg_reference(self.ref_channels)

    def get_raw(self):
        """ Returns the raw data. """
        if self.is_subcortex_:
            return self.raw_, self.vertex_channel
        else:
            return self.raw_


class EEGDownSegmenter():
    """ Class to segment raw EEG data into anti-aliased, downsampled epochs.

    An optional high pass filter can be applied to the data as well. The epochs are corrected for trigger delay.
    An instance of this class directly returns a mne.Epochs object with the data segmented into epochs.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    subject_id : str
        Subject ID
    tmin : float
        Start time of the epoch in seconds
    tmax : float
        End time of the epoch in seconds
    decimator : int
        Decimation factor by which the data is downsampled
    highpass : float | None
        High pass frequency in Hz, if None, no high pass filter is applied
    is_subcortex : bool | False
        Whether the data is for subcortical analysis or not
    is_ABR : bool | False
        Whether the ABR is extracted or not (`is_subcortex` must also be true)

    Returns
    -------
    epochs : mne.Epochs
        Segmented epochs

    Examples
    --------
    >>> eeg_loader = EEGLoader('p01', 'data', '.fif', is_subcortex=False)
    >>> raw = eeg_loader.get_raw()

    """
    def __init__(self, raw, subject_id, tmin, tmax, decimator, highpass=None, is_subcortex=False, is_ABR=False):
        self.raw = raw
        self.subject_id = subject_id
        self.is_subcortex_ = is_subcortex
        self.is_ABR_ = is_ABR
        self.highpass_ = highpass
        self.tmin = tmin
        self.tmax = tmax
        self.decimator = decimator

        self.epochs = None
        self.events_ = None

        if self.is_ABR_ and not self.is_subcortex_:
            raise ValueError('is_ABR can only be true if is_subcortex is true')

        self.create_epochs()

    def anti_aliasing_filter(self):
        """ Applies anti-aliasing filter to the raw data

        An anti-aliasing low pass filter at 1/3 of the target frequency is applied to the raw data, the transition
        width is set to 1/10 of the target frequency. If the data is for subcortical analysis, a causal filter is
        applied, otherwise a zero-phase filter is applied. All frequencies are determined through the decimator
        parameter.

        An optional high pass filter can be applied to the data as well when specified in the constructor.

        """
        sfreq_goal = self.raw.info['sfreq'] / self.decimator

        # Filter specifications
        lowpass = sfreq_goal / 3.0
        l_trans_bandwidth = sfreq_goal / 10.0
        h_trans_bandwidth = None if self.highpass_ is None else self.highpass_ / 4.0
        phase = 'minimum' if self.is_subcortex_ else 'zero'

        self.raw.filter(
            l_freq=self.highpass_,
            l_trans_bandwidth=h_trans_bandwidth,
            h_freq=lowpass,
            h_trans_bandwidth=l_trans_bandwidth,
            method='fir',
            fir_window='hamming',
            phase=phase
        )

    def get_events(self):
        """ Finds audiobook events in the raw data and accounts for trigger delay and participant-related problems.

        The event code for audio onset is "256". The delay from the transductor to the eardrum `delta_t` is 1.07 ms.
        The trigger delay is accounted for by simply adding the delay in samples to the event onset.

        """
        events = mne.find_events(
            self.raw,
            stim_channel='Status',
            min_duration=(1 / self.raw.info['sfreq']),
            shortest_event=1,
            initial_event=True,
        )

        # Special case for participant 07, where three extra events occurred in the break
        if self.subject_id == 'p07' and self.is_ABR_ is False:
            mask = np.ones(events.shape[0], dtype=bool)
            mask[39:42] = False
            events = events[mask]

        delta_t = 1.07e-03
        delay_samples = int(delta_t * self.raw.info['sfreq'])
        mask = events[:, 2] == 256
        audio_events_delta = events[mask, :]
        audio_events_delta[:, 0] = audio_events_delta[:, 0] + delay_samples

        self.events_ = audio_events_delta

    def create_epochs(self):
        self.anti_aliasing_filter()
        self.get_events()

        epochs = mne.Epochs(
            self.raw,
            self.events_,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=None,
            preload=True
        )

        epochs.decimate(self.decimator)

        self.epochs = epochs

    def get_epochs(self):
        return self.epochs


def clean_subcortex_signal(array, sfreq, threshold=100.0, segment_duration=1.0):
    """ Cleans the EEG data by setting segments above a threshold to NaN.

    Parameters
    ----------
    array : numpy array
        EEG data of shape (n_epochs, n_channels, n_samples).
    threshold : float
        Threshold in microvolts.
    sfreq : float
        Sampling frequency in Hz.
    segment_duration : float
        Duration of the segment to zero out in seconds.

    Returns
    -------
    array : numpy array
        Cleaned EEG data of shape (n_epochs, n_channels, n_samples).

    """
    threshold_uV = threshold / 1e6  # convert microvolts to volts
    segment_length = int(sfreq * segment_duration)  # Number of samples in the segment to zero out

    array_cleaned = array.copy()

    for i in range(array_cleaned.shape[0]):

        indices = np.where(np.abs(array_cleaned[i, 0, :]) > threshold_uV)[0]

        for idx in indices:
            start_idx = idx - segment_length // 2
            end_idx = idx + segment_length // 2

            # "Zero" the segment (set to NaN)
            array_cleaned[i, 0, start_idx:end_idx] = np.nan

    return array_cleaned
