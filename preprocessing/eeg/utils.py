import mne

default_subjects = [
    'p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08',
    'p09', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16',
    'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24',
    'p25', 'p26', 'p27', 'p28', 'p29', 'p30', 'p31', 'p32',
    'p33', 'p34', 'p35', 'p36', 'p37', 'p38', 'p39', 'p40',
    'p41', 'p42', 'p44', 'p45'
]

bad_ABR_channels_dict = {
    'p01': ['T8', 'PO4', 'O2', 'F4', 'FC1', 'AF4'],
    'p02': ['T7'],
    'p03': ['AF4', 'Oz', 'F4', 'PO4'],
    'p04': [],
    'p05': [],
    'p06': [],
    'p07': ['FC2'],
    'p08': ['F3'],
    'p09': [],
    'p10': ['Fp2'],
    'p11': [],
    'p12': ['CP5', 'PO4', 'FC2'],
    'p13': ['T7', 'CP2', 'CP6'],
    'p14': ['FC5', 'Fp1', 'T7'],
    'p15': ['Oz', 'Fp2'],
    'p16': [],
    'p17': [],
    'p18': ['T7', 'T8'],
    'p19': [],
    'p20': ['P4', 'Fp1', 'Fp2', 'Pz', 'FC6', 'AF4'],
    'p21': ['Cz'],
    'p22': ['CP5'],
    'p23': ['T7', 'T8'],
    'p24': [],
    'p25': ['F7'],
    'p26': ['C3', 'PO3', 'Fz'],
    'p27': ['P7', 'AF3', 'F7', 'Cz'],
    'p28': [],
    'p29': ['CP6'],
    'p30': ['CP6', 'P8', 'CP5'],
    'p31': ['T7', 'Oz'],
    'p32': [],
    'p33': ['O2', 'P7', 'CP5'],
    'p34': ['P3', 'Cz'],
    'p35': ['EXG1'],
    'p36': ['CP5', 'Pz', 'T7'],
    'p37': [],
    'p38': [],
    'p39': [],
    'p40': ['O2', 'P8', 'CP2'],
    'p41': [],
    'p42': [],
    'p44': [],
    'p45': [],
}

subjects_bad_ABR_refs = ['p04', 'p08', 'p10']

bad_audio_channels_dict = {
    'p01': ['Fp1', 'T8', 'PO4', 'O2', 'FC5'],
    'p02': ['T7'],
    'p03': ['AF4', 'Oz', 'F4', 'PO4'],
    'p04': ['P3', 'CP1'],
    'p05': ['C3', 'P3', 'T7'],
    'p06': ['O1', 'O2', 'P7'],
    'p07': ['Oz', 'FC6', 'PO4'],
    'p08': [],
    'p09': [],
    'p10': ['PO3'],
    'p11': [],
    'p12': ['PO4'],
    'p13': ['CP5', 'T7'],
    'p14': ['Fp1', 'O1'],
    'p15': [],
    'p16': [],
    'p17': [],
    'p18': ['T7', 'T8', 'F8'],
    'p19': ['AF4'],
    'p20': ['Pz', 'O2'],
    'p21': ['Cz'],
    'p22': ['PO3', 'Fz', 'P7'],
    'p23': ['T7', 'T8'],
    'p24': [],
    'p25': [],
    'p26': ['Fz', 'C3', 'P7'],
    'p27': ['P7', 'O1'],
    'p28': [],
    'p29': [],
    'p30': [],
    'p31': ['T7'],
    'p32': ['T8'],
    'p33': [],
    'p34': ['T7', 'Cz'],
    'p35': ['F4', 'EXG1'],
    'p36': [],
    'p37': [],
    'p38': ['FC5', 'Cz'],
    'p39': [],
    'p40': ['P4', 'Oz'],
    'p41': [],
    'p42': [],
    'p44': [],
    'p45': [],
}

subjects_bad_audio_refs = ['p03', 'p11', 'p21', 'p22', 'p38', 'p39']


def parse_arguments(default_subjects):
    """ Argument parser for parsing from command-line.

    """
    import argparse

    parser = argparse.ArgumentParser(description='Process EEG for subcortex analysis')
    parser.add_argument('-s', '--subjects', type=str, help='Subject numbers separated by commas (no spaces)')
    args = parser.parse_args()

    if args.subjects:
        subjects = args.subjects.split(',')
        subjects = [subject.strip() for subject in subjects]
    else:
        subjects = default_subjects

    return subjects

# ICA configuration
ica = mne.preprocessing.ICA(
    n_components=0.999,
    method='picard',
    max_iter=1000,
    fit_params=dict(fastica_it=5),
    random_state=1606
)

# ICA components I decided to remove
IC_dict = {
    'p01': [0, 2, 8],
    'p02': [0, 4],
    'p03': [0, 6],
    'p04': [0, 1, 2, 7],
    'p05': [0, 6],
    'p06': [0, 13],
    'p07': [0, 17],
    'p08': [0, 14],
    'p09': [0, 2, 12],
    'p10': [0, 12],
    'p11': [0, 1, 7],
    'p12': [0, 6, 8],
    'p13': [0, 4],
    'p14': [0, 2],
    'p15': [0, 2],
    'p16': [0, 1, 5],
    'p17': [0, 22],
    'p18': [0, 3],
    'p19': [0, 1, 3],
    'p20': [0, 1, 6],
    'p21': [0, 9],
    'p22': [0, 1, 6],
    'p23': [0],
    'p24': [0, 6],
    'p25': [0, 4, 15],
    'p26': [0],
    'p27': [0, 5],
    'p28': [0, 2, 10],
    'p29': [0, 1, 2, 8],
    'p30': [0, 1],
    'p31': [0, 1, 2],
    'p32': [0, 1],
    'p33': [0, 8],
    'p34': [0, 1, 4],
    'p35': [0, 1, 8],
    'p36': [0, 2],
    'p37': [0, 6],
    'p38': [0, 1, 22],
    'p39': [0, 1, 14],
    'p40': [0, 1, 2, 4, 5],
    'p41': [0, 1, 4, 5],
    'p42': [0, 3],
    'p44': [0, 8],
    'p45': [0, 7],
}

channel_names = [
    'Fp1',
    'AF3',
    'F7',
    'F3',
    'FC1',
    'FC5',
    'T7',
    'C3',
    'CP1',
    'CP5',
    'P7',
    'P3',
    'Pz',
    'PO3',
    'O1',
    'Oz',
    'O2',
    'PO4',
    'P4',
    'P8',
    'CP6',
    'CP2',
    'C4',
    'T8',
    'FC6',
    'FC2',
    'F4',
    'F8',
    'AF4',
    'Fp2',
    'Fz',
    'Cz'
]