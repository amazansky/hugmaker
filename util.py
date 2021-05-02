import os
import yaml
import cv2 as cv

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
    'flux': 'genderflux',
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

# define a rotate function to rotate the flag for person 2
def rotate(img, angle, scale=1.0):
    (height,width) = img.shape[:2]
    point = (width // 2, height // 2)
    matrix = cv.getRotationMatrix2D(point, angle, scale)
    dimensions = (width, height)
    return cv.warpAffine(img, matrix, dimensions)
