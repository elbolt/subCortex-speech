import envelopes
import an_rates
from pathlib import Path

import mne
mne.set_log_level('WARNING')


if __name__ == '__main__':
    """" Run pipelines to extract envelope and speech features from the audiobook segments.

    Envelopes are directly extracted from the audio files, while speech features are extracted from the output of
    the AN model. AN rates are created twice, they were once extracted from the original speech wave, and once from the
    inverted speech wave. This step of the pipeline is documented in the subfolder `an_model`.

    """
    print('Running: ', __file__)

    audiobook_segments = [
        'snip01', 'snip02', 'snip03', 'snip04', 'snip05', 'snip06', 'snip07', 'snip08', 'snip09', 'snip10',
        'snip11', 'snip12', 'snip13', 'snip14', 'snip15', 'snip16', 'snip17', 'snip18', 'snip19', 'snip20',
        'snip21', 'snip22', 'snip23', 'snip24', 'snip25'
    ]

    # Path to my `audiofiles` folder
    SSD_dir = Path('/Volumes/NeuroSSD/Midcortex/data/audiofiles')

    # Path to the folders containing the wav files, TextGrids, and output of the AN model
    wav_folder = SSD_dir / 'raw'
    an_folder = SSD_dir / 'Zilany_2014' / 'anm_features'
    an_folder_invered = SSD_dir / 'Zilany_2014' / 'anm_features_inverted'
    tg_folder = SSD_dir / 'textgrids'
    out_folder = SSD_dir / 'features'

    # Path to out folder where processed data will be stored
    out_folder.mkdir(parents=True, exist_ok=True)

    envelopes.create(wav_folder, tg_folder, out_folder, audiobook_segments)
    an_rates.create(an_folder, tg_folder, out_folder, audiobook_segments)
    an_rates.create(an_folder_invered, tg_folder, out_folder, audiobook_segments, is_inverted=True)
