from encoder_utils import FeatureTargetLoader, Preparer, Encoder


def run_subcortical(subject, directory, alphas, invert=False):
    """ Runs the subcortical encoding model.

    The data is loaded using the FeatureTargetLoader class, prepared using the Preparer class, and then encoded using
    the Encoder class. The Encoder class is a wrapper for the scikit-learn RidgeCV class.

    Parameters
    ----------
    subject : str
        Subject ID.
    directory : str
        directory where the data is stored.
    alphas : list
        List of alphas to perform grid search over.
    invert : bool | False
        Whether this is the model using speech features retrieved from inverted speech waves.

    Returns
    -------
    response : np.ndarray
        Response of the model.
    test_score : float
        Score of the model.
    lags : np.ndarray
        Lags of the model.
    best_alpha : float
        Best alpha found by hyperparameter tuning.

    """
    # Subcortical encoding parameters
    tmin, tmax = -10e-3, 30e-3
    sfreq = 4096

    loader = FeatureTargetLoader(
        subject,
        directory,
        feature_type='an_rates',
        is_subcortex=True
    )
    feature, feature_invert, eeg = loader.get_data()

    feature = feature_invert if invert else feature

    preparer = Preparer(subject, feature, eeg, is_subcortex=True)

    feature_train, feature_test, eeg_train, eeg_test = preparer.get_data()

    encoder = Encoder(
        feature_train,
        feature_test,
        eeg_train,
        eeg_test,
        tmin=tmin,
        tmax=tmax,
        sfreq=sfreq
    )

    best_alpha, alpha_scores = encoder.tune_alpha(alphas=alphas)

    # Fit model and get test score
    encoder.fit(alpha=best_alpha)

    response, test_score, lags = encoder.get_data()

    return response, test_score, alpha_scores, best_alpha, lags


def run_cortical(subject, directory, alphas):
    """
    Runs the subcortical encoding model.

    Again, the data is loaded using the FeatureTargetLoader class, prepared using the Preparer class, and then encoded
    using the Encoder class.


    Parameters
    ----------
    subject : str
        Subject ID.
    directory : str
        directory where the data is stored.
    alphas : list
        List of alphas to perform grid search over.


    Returns
    -------
    response : np.ndarray
        Response of the model.
    test_score : float
        Score of the model.
    lags : np.ndarray
        Lags of the model.
    best_alpha : float
        Best alpha found by hyperparameter tuning.

    """
    # Cortical encoding parameters
    tmin, tmax = -300e-3, 600e-3
    sfreq = 128

    loader = FeatureTargetLoader(
        subject,
        directory,
        feature_type='envelope'
    )
    feature, eeg = loader.get_data()

    preparer = Preparer(
        subject,
        feature,
        eeg
    )
    feature_train, feature_test, eeg_train, eeg_test = preparer.get_data()

    encoder = Encoder(
        feature_train,
        feature_test,
        eeg_train,
        eeg_test,
        tmin=tmin,
        tmax=tmax,
        sfreq=sfreq
    )

    best_alpha, alpha_scores = encoder.tune_alpha(alphas=alphas)

    encoder.fit(alpha=best_alpha)

    response, test_score, lags = encoder.get_data()

    return response, test_score, lags, best_alpha, alpha_scores
