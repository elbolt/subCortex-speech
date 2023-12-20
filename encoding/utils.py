default_subjects = [
    'p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08',
    'p09', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16',
    'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24',
    'p25', 'p26', 'p27', 'p28', 'p29', 'p30', 'p31', 'p32',
    'p33', 'p34', 'p35', 'p36', 'p37', 'p38', 'p39', 'p40',
    'p41', 'p42', 'p44', 'p45'
]


def parse_arguments(default_subjects):
    """ Argument parser for parsing from command-line.

    """
    import argparse

    parser = argparse.ArgumentParser(description='Parsing subjects for encoding analysis.')
    parser.add_argument('-s', '--subjects', type=str, help='Subject numbers separated by commas (no spaces)')
    args = parser.parse_args()

    if args.subjects:
        subjects = args.subjects.split(',')
        subjects = [subject.strip() for subject in subjects]
    else:
        subjects = default_subjects

    return subjects
