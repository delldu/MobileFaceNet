"""Mobile Face Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#
__vesion__ = "1.0.0"

import torch
from torchvision import transforms as T
from PIL import ImageDraw
import numpy as np

from . import mobileface
from . import retinaface

# reference facial points, a list of coordinates (x,y)
REFERENCE_FACE_POINTS = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156],
]
# (width, height)
REFERENCE_FACE_SIZE = (96, 112)

# standard facial points, a list of coordinates (x,y)
STANDARD_FACE_POINTS = [
    [38.29459953, 51.69630051],
    [73.53179932, 51.50139999],
    [56.02519989, 71.73660278],
    [41.54930115, 92.3655014],
    [70.72990036, 92.20410156],
]
# (width, height)
STANDARD_FACE_SIZE = (112, 112)


def tensor(image):
    """image is Pillow image object, return 1xCxHxW tensor"""
    t = T.Compose(
        [
            T.ToTensor(),
        ]
    )
    return t(image).unsqueeze(0)


def detector(device=torch.device("cuda")):
    return retinaface.Detector(device)


def extractor(device=torch.device("cuda")):
    return mobileface.Extractor(device)


def draw(image, dets, landms):
    draw = ImageDraw.Draw(image)
    for b in dets:
        # box = (x1, y1, x2, y2)
        box = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        draw.rectangle(box, fill=None, outline="#FFFFFF", width=1)

    for p in landms:
        for i in range(5):
            # box = (x1, y1, x2, y2)
            box = (int(p[i]) - 1, int(p[i + 5]) - 1, int(p[i]) + 1, int(p[i + 5]) + 1)
            draw.ellipse(box, fill=None, outline="#00FF00", width=1)


def crop(image, dets):
    faces = []
    for b in dets:
        # box = (x1, y1, x2, y2)
        box = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        faces.append(image.crop(box))
    return faces


def align(image, landms):
    faces = []

    eye_distance = STANDARD_FACE_POINTS[1][0] - STANDARD_FACE_POINTS[0][0]

    mid_mouth_y = (STANDARD_FACE_POINTS[3][1] + STANDARD_FACE_POINTS[4][1]) / 2.0
    mid_eye_y = (STANDARD_FACE_POINTS[0][1] + STANDARD_FACE_POINTS[1][1]) / 2.0
    eye_to_mouth_distance = mid_mouth_y - mid_eye_y

    half_face_width = STANDARD_FACE_SIZE[0] / 2.0
    half_face_height = STANDARD_FACE_SIZE[1] / 2.0

    for landmark in landms:
        points = landmark.reshape(2, 5).T

        # center
        center_x, center_y = points.mean(axis=0)

        # theta
        eye_dx = points[1][0] - points[0][0]
        eye_dy = points[1][1] - points[0][1]
        theta = np.degrees(np.arctan2(eye_dy, eye_dx))

        # scale x/y
        scale_x = eye_dx / eye_distance
        mouth_cy = (points[3][1] + points[4][1]) / 2.0
        eye_cy = (points[1][1] + points[0][1]) / 2.0
        scale_y = (mouth_cy - eye_cy) / eye_to_mouth_distance

        # Fast align: first crop, then rotate
        # Suppose face height < 4 * (mout_cy - eye_cy), width < 4 * eye_cy
        left = max(center_x - 2 * eye_dx, 0)
        top = max(center_y - 2 * (mouth_cy - eye_cy), 0)
        right = min(center_x + 2 * eye_dx, image.width)
        bottom = min(center_y + 2 * (mouth_cy - eye_cy), image.height)
        crop_box = (left, top, right, bottom)
        nimage = image.crop(crop_box)
        # new center
        center_x -= left
        center_y -= top
        # rotate
        nimage = nimage.rotate(theta, center=(center_x, center_y))

        # final crop to standard size: STANDARD_FACE_SIZE
        crop_box = (
            center_x - half_face_width * scale_x,
            center_y - half_face_height * scale_y,
            center_x + half_face_width * scale_x,
            center_y + half_face_height * scale_y,
        )
        faces.append(nimage.crop(crop_box).resize(STANDARD_FACE_SIZE))
    return faces
