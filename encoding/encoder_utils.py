import numpy as np
from collections import defaultdict
from mne.decoding import TimeDelayingRidge, ReceptiveField
from sklearn.model_selection import train_test_split, LeaveOneOut
from tqdm.auto import tqdm

import mne
mne.set_log_level('WARNING')


class FeatureTargetLoader():
    """ Class to load feature (speech feature) and target (EEG) data for a given subject and analysis type.

    Note: The paths in this class are very specific to my own data structure and would need a lot of adaption.

    Methods
    -------
    set_file_names()
        Sets the file names of the feature and EEG data.
    check_arguments()
        Checks whether the arguments and paths are valid.
    load_data()
        Loads the feature and EEG data.
    get_data()
        Returns the feature and EEG data.

    """
    def __init__(
        self,
        subject_id: str,
        directory: str,
        feature_type: str = 'envelope',
        is_subcortex: bool = False
    ) -> None:
        """ Initialize the FeatureTargetLoader class.

        Parameters
        ----------
        subject_id : str
            Subject ID.
        directory : pathlib.Path
            Path to the data directory.
        feature_type : str | 'envelope'
            Type of feature to load. Can be `envelope`, `an_rates` or `an_rates_cortical`.
        is_subcortex : bool | False
            Whether to use subcortical data or not.

        """
        self.subject_id = subject_id
        self.directory = directory
        self.feature_type = feature_type
        self.is_subcortex = is_subcortex

        self.set_file_names()
        self.check_arguments()
        self.load_data()

    def set_file_names(self):
        """ Sets the file names of the feature and EEG data.

        Preprocessed EEG files need to be of shape (n_epochs, n_channels, n_times) and stored in a directory such as:

        EEG/
            TRF/
                preprocessed/
                    subcortex/
                        p01_subcortex.npy
                        p02_subcortex.npy
                        ...
                    cortex/
                        p01_cortex.npy
                        p02_cortex.npy
                        ...

        Preprocessed features need be of shape (n_epochs, 1, n_times) and stored in a directory such as:

        audiofiles/
            features/
                envelopes.npy
                an_rates.npy
                an_rates_inverted.npy

        """
        EEG_dir = 'EEG/TRF/preprocessed/'

        if self.is_subcortex:
            filename = f'subcortex/{self.subject_id}_subcortex.npy'
        elif not self.is_subcortex:
            filename = f'cortex/{self.subject_id}_cortex.npy'
        self.eeg_file = self.directory / EEG_dir / filename

        feature_dir = 'audiofiles/features'

        if self.feature_type == 'envelope':
            filename_normal = 'envelopes.npy'
            self.feature_file = self.directory / feature_dir / filename_normal

        elif self.feature_type == 'an_rates':
            filename_normal = 'an_rates.npy'
            filename_invert = 'an_rates_inverted.npy'
            self.feature_file = self.directory / feature_dir / filename_normal
            self.feature_file_invert = self.directory / feature_dir / filename_invert

        elif self.feature_type == 'an_rates_cortical':
            filename_normal = 'an_rates_cortical.npy'
            self.feature_file = self.directory / feature_dir / filename_normal

    def check_arguments(self):
        """ Checks whether the arguments and paths are valid.

        """
        if self.feature_type not in ['envelope', 'an_rates', 'an_rates_cortical']:
            raise ValueError('feature must be `envelope` or `an_rates`.')
        if self.is_subcortex and self.feature_type == 'envelope':
            raise ValueError('is_subcortex can only be True if feature is `an_rates`.')
        elif not self.is_subcortex and self.feature_type != 'envelope' and self.feature_type != 'an_rates_cortical':
            raise ValueError('`envelope` or `an_rates_cortical` must be used for cortical data.')

        if self.directory.is_dir() is False:
            raise ValueError(f'File {self.eeg_file} does not exist.')
        if self.feature_file.is_file() is False:
            raise ValueError(f'File {self.feature_file} does not exist.')

    def load_data(self):
        """ Loads the feature and EEG data.

        """
        self.feature = np.load(self.feature_file)
        self.eeg = np.load(self.eeg_file)

        if self.feature_type == 'an_rates':
            self.feature_invert = np.load(self.feature_file_invert)

        # Remove first feature trial for participant p45 (forgot to record right away)
        if self.subject_id == 'p45':
            self.feature = np.delete(self.feature, 0, axis=0)

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """ Returns the feature and EEG data.

        Returns
        -------
        feature : np.ndarray
            feature data. If feature_type is `an_rates`, returns two features.
        eeg : np.ndarray
            EEG data.

        """
        if self.feature_type == 'an_rates':
            return self.feature, self.feature_invert, self.eeg
        else:
            return self.feature, self.eeg


class Preparer():
    """ Class to prepare feature (speech feature) and target (EEG) data for a given subject and analysis type.

    Methods
    -------
    reshape_data()
        Reshapes the data.
    get_train_test_set()
        Split data into train/test sets.
    normalize_eeg()
        Applies global normalization (z-scoring) to the eeg data.
    normalize_feature()
        Applies global normalization (z-scoring) to the feature data.
    get_data()
        Returns the feature and EEG data.

    """
    def __init__(self, subject_id: str, feature: np.ndarray, eeg: np.ndarray, is_subcortex: bool = False) -> None:
        """ Initialize the Preparer class.

        Parameters
        ----------
        subject_id : str
            Subject ID.
        feature : np.ndarray
            feature data.
        eeg : np.ndarray
            EEG data.
        is_subcortex : bool | False
            Whether to use subcortical data or not.

        """
        self.feature = feature
        self.eeg = eeg
        self.is_subcortex = is_subcortex
        self.n_epochs = self.eeg.shape[0]
        self.seed = int(subject_id[-2:]) if is_subcortex else int(subject_id[-2:]) + 2023

        self.reshape_data()
        self.normalize_eeg()
        self.get_train_test_set()
        self.normalize_feature()

    def reshape_data(self) -> None:
        """ Reshapes the data.

        mne.Epochs.get_data() retruns array of shape (n_epochs, n_channels, n_times);
        mne.decoding.ReceptiveField.fit() expects array of shape (n_times, n_epochs, n_features).

        """
        self.feature = np.transpose(self.feature, (2, 0, 1))
        self.eeg = np.transpose(self.eeg, (2, 0, 1))

    def get_train_test_set(self) -> None:
        """ Split data into train/test sets using a seed for reproducibility.

        """
        epoch_indices = np.arange(self.n_epochs)

        train_indices, test_indices = train_test_split(
            epoch_indices,
            test_size=3,
            random_state=self.seed
        )

        self.feature_train = self.feature[:, train_indices, :]
        self.feature_test = self.feature[:, test_indices, :]
        self.eeg_train = self.eeg[:, train_indices, :]
        self.eeg_test = self.eeg[:, test_indices, :]

    def normalize_eeg(self) -> None:
        """ Applies global normalization (z-scoring) to the eeg data and replaces NaN values with 0.0 in subcortical
        data.

        """
        eeg = (self.eeg - np.nanmean(self.eeg)) / np.nanstd(self.eeg)
        if self.is_subcortex:
            eeg = np.nan_to_num(eeg, nan=0.0)

        self.eeg = eeg

    def normalize_feature(self) -> None:
        """ Applies global normalization (z-scoring) to the feature data and replaces NaN values with 0.0.

        The test data are normalized using the mean and standard deviation of the train data.

        """
        train_mean = np.nanmean(self.feature_train)
        train_std = np.nanstd(self.feature_train)

        feature_train = (self.feature_train - train_mean) / train_std
        feature_test = (self.feature_test - train_mean) / train_std

        self.feature_train = np.nan_to_num(feature_train, nan=0.0)
        self.feature_test = np.nan_to_num(feature_test, nan=0.0)

    def get_data(self):
        return self.feature_train, self.feature_test, self.eeg_train, self.eeg_test


class Encoder():
    """ Class to run the neural encoding model pipeline.

    Methods
    -------
    tune_alpha()
        Tune alpha parameter for TimeDelayingRidge estimator.
    fit()
        Fit the model using a given alpha parameter.
    get_data()
        Returns the response, test score, and lags of the model.

    """
    def __init__(
        self,
        feature_train: np.ndarray,
        feature_test: np.ndarray,
        eeg_train: np.ndarray,
        eeg_test: np.ndarray,
        tmin: float,
        tmax: float,
        sfreq: float
    ) -> None:
        """ Initialize the Encoder class.

        Parameters
        ----------
        feature_train : np.ndarray
            feature data for training.
        feature_test : np.ndarray
            feature data for testing.
        eeg_train : np.ndarray
            EEG data for training.
        eeg_test : np.ndarray
            EEG data for testing.
        tmin : float
            Start time of the receptive field.
        tmax : float
            End time of the receptive field.
        sfreq : float
            Sampling frequency of the data.

        """
        self.tmin = tmin
        self.tmax = tmax
        self.sfreq = sfreq

        self.feature_train = feature_train
        self.feature_test = feature_test
        self.eeg_train = eeg_train
        self.eeg_test = eeg_test

    def tune_alpha(
        self,
        alphas: list,
        tmin: float = None,
        tmax: float = None,
        sfreq: float = None
    ) -> tuple[float, dict]:
        """ Tune alpha parameter for TimeDelayingRidge estimator.

        The method applies leave-one-out cross-validation to the eeg epochs left in the training data.
        Time lags for tuning can be defined, else those of the class instance are used.

        Parameters
        ----------
        alphas : list
            List of alpha parameters to tune.
        tmin : float | None
            Start time of the receptive field.
        tmax : float | None
            End time of the receptive field.
        sfreq : float | None
            Sampling frequency of the data.

        Returns
        -------
        best_alpha : float
            Best alpha parameter found by hyperparameter tuning.
        scores : dict
            Dictionary of scores for each alpha.

        """
        tmin = self.tmin if tmin is None else tmin
        tmax = self.tmax if tmax is None else tmax
        sfreq = self.sfreq if sfreq is None else sfreq

        loo = LeaveOneOut()
        scores = defaultdict(float)
        best_score = -np.inf
        best_alpha = None
        scores = defaultdict(float)
        folds = self.eeg_train.shape[1]

        for alpha in alphas:
            desc = f'alpha {alpha:16.3f}'

            for train_indices, validation_index in tqdm(loo.split(np.arange(folds)), total=folds, desc=desc):
                feature_train = self.feature_train[:, train_indices, :]
                feature_validation = self.feature_train[:, validation_index, :]
                eeg_train = self.eeg_train[:, train_indices, :]
                eeg_validation = self.eeg_train[:, validation_index, :]

                estimator = TimeDelayingRidge(
                    tmin=tmin,
                    tmax=tmax,
                    sfreq=sfreq,
                    reg_type='laplacian',
                    alpha=alpha
                )

                model = ReceptiveField(
                    tmin=tmin,
                    tmax=tmax,
                    estimator=estimator,
                    sfreq=sfreq,
                    scoring='corrcoef'
                )

                model.fit(feature_train, eeg_train)

                score = model.score(feature_validation, eeg_validation)
                score = score.mean()

                scores[alpha] += score

            scores[alpha] /= folds

            if self.sfreq == 4096:
                scores[alpha] = np.round(scores[alpha], 4)
            else:
                scores[alpha] = np.round(scores[alpha], 3)

            if scores[alpha] >= best_score:
                best_score = scores[alpha]
                best_alpha = alpha

        return best_alpha, scores

    def fit(self, alpha: float):
        """ Fit the model using a given alpha parameter. """
        estimator = TimeDelayingRidge(
            tmin=self.tmin,
            tmax=self.tmax,
            sfreq=self.sfreq,
            reg_type='laplacian',
            alpha=alpha
        )

        model = ReceptiveField(
            tmin=self.tmin,
            tmax=self.tmax,
            estimator=estimator,
            sfreq=self.sfreq,
            scoring='corrcoef'
        )

        model.fit(self.feature_train, self.eeg_train)

        score = model.score(self.feature_test, self.eeg_test).mean()

        response = model.coef_.squeeze()

        self.lags = model.delays_ / model.sfreq * 1e3

        self.response = response
        self.test_score = score

    def get_data(self) -> tuple[np.ndarray, float, np.ndarray]:
        """ Returns the response, test score, and lags of the model. """
        return self.response, self.test_score, self.lags
