import cv2 as cv
import numpy as np

img = cv.imread('images/hug.png')

# flag colors use BGR for consistency with OpenCV
flags = {
    'ace': [(0, 0, 0), (163, 163, 163), (255, 255, 255), (128, 0, 128)],
    'agender': [(0, 0, 0), (185, 185, 185), (255, 255, 255), (131, 244, 184), (255, 255, 255), (185, 185, 185), (0, 0, 0)],
    'aro': [(66, 165, 61), (121, 211, 167), (255, 255, 255), (169, 169, 169), (0, 0, 0)],
    'bi': [(112, 2, 214), (112, 2, 214), (150, 79, 155), (168, 56, 0), (168, 56, 0)],
    'bigender': [(157, 122, 193), (200, 165, 234), (229, 198, 212), (255, 255, 255), (229, 198, 212), (230, 198, 155), (203, 130, 107)],
    'enby': [(48, 244, 255), (255, 255, 255), (209, 89, 156), (0, 0, 0)],
    'gay': [(113, 141, 19), (171, 206, 45), (195, 232, 153), (255, 255, 255), (223, 173, 124), (199, 73, 79), (119, 28, 61)],
    'genderfluid': [(162, 117, 255), (255, 255, 255), (214, 24, 190), (0, 0, 0), (189, 62, 51)],
    'genderqueer': [(220, 126, 181), (255, 255, 255), (35, 129, 74)],
    'lesbian': [(0, 45, 213), (86, 154, 255), (255, 255, 255), (164, 98, 211), (83, 2, 138)],
    'omni': [(201, 153, 253), (185, 85, 254), (69, 3, 40), (248, 94, 100), (250, 163, 138)],
    'pan': [(140, 33, 255), (0, 216, 255), (255, 177, 33)],
    'poly': [(185, 28, 246), (105, 213, 7), (246, 146, 28)],
    'pride': [(3, 3, 228), (0, 140, 255), (0, 237, 255), (38, 128, 0), (255, 77, 0), (135, 7, 117)],
    'trans': [(250, 206, 91), (184, 169, 245), (255, 255, 255), (184, 169, 245), (250, 206, 91)]
}

# make a mask for each of the people in the hug
lightblue = np.array([238, 172, 85], dtype = 'uint16')
darkblue = np.array([153, 102, 34], dtype = 'uint16')

mask1 = cv.inRange(img, darkblue, darkblue)
mask2 = cv.inRange(img, lightblue, lightblue)

# change these to whatever flags you want from the list above
selected1 = 'pride'
selected2 = 'pride'

# define a rotate function to rotate the right flag
def rotate(img, angle, scale=1.0, rotPoint=None):
    (height,width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width // 2, height // 2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, scale)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)

# create each flag from its corresponding colors
flag1 = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
colors1 = flags[selected1]
print(len(colors1))
for i, color in enumerate(colors1):
    cv.rectangle(flag1, (0, i*img.shape[1] // len(colors1)), (img.shape[1], (i+1) * img.shape[1] // len(colors1)), color, -1)

flag2 = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
colors2 = flags[selected2]
for i, color in enumerate(colors2):
    cv.rectangle(flag2, (0, i*img.shape[1] // len(colors2)), (img.shape[1], (i+1) * img.shape[1] // len(colors2)), color, -1)
flag2 = rotate(flag2, 5, 1.1)

# use the people as masks for the flags
person1 = cv.bitwise_and(flag1, flag1, mask=mask1)
# blurred1 = cv.GaussianBlur(person1, (5, 5), cv.BORDER_DEFAULT)
person2 = cv.bitwise_and(flag2, flag2, mask=mask2)
# blurred2 = cv.GaussianBlur(person2, (5, 5), cv.BORDER_DEFAULT)

people = cv.bitwise_or(person1, person2)
# people = cv.bitwise_or(blurred1, blurred2)

cv.imshow('final', people)

cv.waitKey(0)
