import cv2 as cv
import numpy as np

# flag colors use BGR (not RGB!) for consistency with OpenCV
flagdict = {
    'abrosexual': [(149, 202, 119), (202, 228, 181), (255, 255, 255), (181, 150, 230), (110, 70, 216)],
    'asexual': [(0, 0, 0), (163, 163, 163), (255, 255, 255), (128, 0, 128)],
    'agender': [(0, 0, 0), (185, 185, 185), (255, 255, 255), (131, 244, 184), (255, 255, 255), (185, 185, 185), (0, 0, 0)],
    'aroace': [(42, 140, 225), (57, 205, 236), (255, 255, 255), (217, 174, 99), (85, 56, 32)],
    'aromantic': [(66, 165, 61), (121, 211, 167), (255, 255, 255), (169, 169, 169), (0, 0, 0)],
    'bigender': [(157, 122, 193), (200, 165, 234), (229, 198, 212), (255, 255, 255), (229, 198, 212), (230, 198, 155), (203, 130, 107)],
    'bisexual': [(112, 2, 214), (112, 2, 214), (150, 79, 155), (168, 56, 0), (168, 56, 0)],
    'boyflux': [(248, 233, 214), (244, 171, 111), (111, 54, 2), (171, 237, 158), (111, 54, 2), (244, 171, 111), (248, 233, 214)],
    'demiboy': [(127, 127, 127), (195, 195, 195), (233, 217, 154), (255, 255, 255), (233, 217, 154), (195, 195, 195), (127, 127, 127)],
    'demigirl': [(127, 127, 127), (195, 195, 195), (200, 175, 254), (255, 255, 255), (200, 175, 254), (195, 195, 195), (127, 127, 127)],
    'gay': [(113, 141, 19), (171, 206, 45), (195, 232, 153), (255, 255, 255), (223, 173, 124), (199, 73, 79), (119, 28, 61)],
    'genderfluid': [(162, 117, 255), (255, 255, 255), (214, 24, 190), (0, 0, 0), (189, 62, 51)],
    'genderflux': [(148, 119, 243), (185, 163, 241), (206, 206, 206), (245, 224, 126), (246, 205, 66), (149, 244, 255)],
    'genderqueer': [(220, 126, 181), (255, 255, 255), (35, 129, 74)],
    'girlflux': [(215, 230, 249), (108, 82, 242), (17, 3, 191), (135, 197, 233), (17, 3, 191), (108, 82, 242), (215, 230, 249)],
    'graysexual': [(143, 12, 115), (169, 176, 173), (255, 255, 255), (169, 176, 173), (143, 12, 115)],
    'grayromantic': [(37, 125, 17), (175, 178, 176), (255, 255, 255), (175, 178, 176), (37, 125, 17)],
    'lesbian': [(0, 45, 213), (86, 154, 255), (255, 255, 255), (164, 98, 211), (83, 2, 138)],
    'neptunic': [(184, 155, 19), (199, 213, 63), (212, 233, 117), (238, 231, 162), (231, 176, 154), (234, 152, 150)],
    'nonbinary': [(48, 244, 255), (255, 255, 255), (209, 89, 156), (0, 0, 0)],
    'omnisexual': [(201, 153, 253), (185, 85, 254), (69, 3, 40), (248, 94, 100), (250, 163, 138)],
    'pansexual': [(140, 33, 255), (0, 216, 255), (255, 177, 33)],
    'pangender': [(159, 247, 255), (206, 221, 255), (250, 235, 255), (255, 255, 255), (250, 235, 255), (206, 221, 255), (159, 247, 255)],
    'polysexual': [(185, 28, 246), (105, 213, 7), (246, 146, 28)],
    'pride': [(3, 3, 228), (0, 140, 255), (0, 237, 255), (38, 128, 0), (255, 77, 0), (135, 7, 117)],
    'toric': [(201, 83, 179), (219, 160, 220), (238, 238, 175), (159, 251, 154), (129, 205, 126)],
    'transfeminine': [(197, 129, 251), (96, 26, 187), (253, 197, 250), (96, 26, 187), (197, 129, 251)],
    'transgender': [(250, 206, 91), (184, 169, 245), (255, 255, 255), (184, 169, 245), (250, 206, 91)],
    'transmasculine': [(228, 162, 17), (147, 4, 27), (253, 244, 193), (147, 4, 27), (228, 162, 17)],
    'trixic': [(205, 82, 180), (237, 173, 238), (203, 192, 255), (185, 231, 255), (15, 185, 255)],
    'uranic': [(210, 123, 60), (236, 161, 80), (248, 201, 106), (251, 235, 155), (224, 251, 255), (201, 242, 255)],
}

for flag in flagdict:
    size = 2048
    blank = np.zeros((size, size, 4), dtype=np.uint8)

    colors = flagdict[flag]

    for i, color in enumerate(colors):
        color = list(color)
        color.append(255)
        color = tuple(color)

        cv.rectangle(
            blank,
            (0, i*size // len(colors)),
            (size, (i+1) * size // len(colors)),
            color,
            -1
        )

    cv.imwrite(f'../flags/{flag}.png', blank)
