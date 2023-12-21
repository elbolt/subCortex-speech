""" Run Zilany et al. (2014) model.

This module contains functions for running the Zilany et al. (2014) model for a given pressure wave and characteristic
frequency to get the auditory nerve (AN) firing rates.
The pressure wave is read in as a .npy file, and the AN firing rates are saved as a .npy file.

This needs to be run in Python 2.7 with numpy 1.15.4 to work with the Zilany model.

"""
import os
import numpy as np
import cochlea


def get_an_rates(pressure_wave, sfreq, cfs):
    """ Run the Zilany et al. 2014 model for a given pressure wave and characteristic frequency to get the auditory
    nerve (AN) firing rates.

    Parameters
    ----------
    pressure_wave : array_like
        The pressure wave to be used as input to the model.
    sfreq : float
        The sampling frequency of the pressure wave.
    cf : float
        The characteristic frequency of the auditory nerve fiber to be simulated.

    Returns
    -------
    an_rates : array_like
        The auditory nerve firing rates for the given pressure wave and characteristic frequency.

    """
    an_rates = cochlea.run_zilany2014_rate(
        pressure_wave,
        sfreq,
        anf_types='hsr',
        cf=cfs,
        species='human',
        cohc=1,
        cihc=1,
        powerlaw='approximate'
    )

    return np.array(an_rates)


if __name__ == '__main__':
    from utils import audiobook_segments, parser

    args = parser.parse_args()

    # Path to my `Zilany_2014` folder
    directory = '/Volumes/NeuroSSD/subCortex-speech/data/audiofiles/Zilany_2014'

    # Path to outpt folders configuration
    if args.input_type == 'inverted':
        folder_in = os.path.join(directory, 'input_arrays_inverted')
        folder_out = os.path.join(directory, 'an_rate_arrays_inverted')
    else:
        folder_in = os.path.join(directory, 'input_arrays')
        folder_out = os.path.join(directory, 'an_rate_arrays')

    # Signal parameters
    SFREQ_COCHLEA = 100e3  # sfreq for model

    cfs = np.logspace(  # characteristic frequencies `cfs`
        np.log10(125),
        np.log10(16e3),
        num=43,
        endpoint=True,
        base=10
    )

    for snip_id in audiobook_segments:
        print('Processing {}'.format(snip_id))

        pressure_wave = np.load(os.path.join(folder_in, '{}_100k_dB.npy'.format(snip_id)))

        an_rates = get_an_rates(pressure_wave, SFREQ_COCHLEA, cfs)

        filename = os.path.join(folder_out, '{}_an_rates_100k.npy'.format(snip_id))

        np.save(filename, an_rates)
