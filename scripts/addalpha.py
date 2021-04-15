#!/usr/bin/env python3

# This is a supplemental command-line utility to be used with hugmaker. It will add an alpha (transparency) layer to
# PNG images. This is helpful in cases where you would like to add an external image as a flag but it does not already
# have an alpha layer, meaning that it will not work with the bot. Note that if the image already has an alpha layer,
# the program will replace the values with the set value (or 255 by default), so be careful.

import argparse
import cv2 as cv

parser = argparse.ArgumentParser(description='Add an alpha channel to one or multiple PNG file(s).')

parser.add_argument('path', nargs='+', type=str, help='the path(s) of the images to convert')
parser.add_argument('-c', '--confirm', action='store_true', help='auto-confirm file overwrites')
parser.add_argument('-s', '--suffix', default='', type=str, help='suffix to append to written image paths')
parser.add_argument('-t', '--transparency', default=255, type=int, help='transparency value for the alpha layer, 0-255')

args = parser.parse_args()

conf = 'This script will overwrite the selected file(s) if you did not specify a suffix. Are you sure you wish to \
continue? (y/N) '

if args.confirm or input(conf).lower() == 'y': # make sure the user confirms file overwrites
    for img_path in args.path:
        rgb = cv.imread(img_path)

        try:
            rgba = cv.cvtColor(rgb, cv.COLOR_BGR2BGRA)
        except cv.error:
            parser.error(f'File {img_path} could not be read successfully. Are you sure it exists and is an image?')
        
        if args.transparency < 0 or args.transparency > 255:
            parser.warning('Transparency set to an invalid value. The image may not output correctly.')
        rgba[:, :, 3] = args.transparency
        
        cv.imwrite(img_path + args.suffix, rgba)
    
else:
    print('Aborting.')
