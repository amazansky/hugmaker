import os
import yaml

# generate set of flags from png files in flags folder
flagset = {f[:-4] for f in os.listdir('flags') if f.endswith('.png')}

# define flag aliases
aliases = {
    'abro': 'abrosexual',
    'ace': 'asexual',
    'aro': 'aromantic',
    'bi': 'bisexual',
    'demi': 'demisexual',
    'demiro': 'demiromantic',
    'enby': 'nonbinary',
    'fluid': 'genderfluid',
    'gray': 'graysexual',
    'grayce': 'graysexual',
    'grayro': 'grayromantic',
    'les': 'lesbian',
    'nb': 'nonbinary',
    'omni': 'omnisexual',
    'pan': 'pansexual',
    'poly': 'polysexual',
    'trans': 'transgender'
}

# access config file
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)
