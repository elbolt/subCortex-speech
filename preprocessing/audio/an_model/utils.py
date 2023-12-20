import argparse

audiobook_segments = [
    'snip01', 'snip02', 'snip03', 'snip04', 'snip05', 'snip06', 'snip07', 'snip08', 'snip09', 'snip10',
    'snip11', 'snip12', 'snip13', 'snip14', 'snip15', 'snip16', 'snip17', 'snip18', 'snip19', 'snip20',
    'snip21', 'snip22', 'snip23', 'snip24', 'snip25'
]

parser = argparse.ArgumentParser(description="Prepare input for Zilany model.")
parser.add_argument(
    '-i',
    '--input_type',
    choices=['normal', 'inverted'],
    default='normal',
    help='Specify the input type (normal or inverted, default is normal)'
)
