import cv2
import numpy as np
from random import randint, uniform
import os


def random_crop(img):
    height, width, channels = img.shape
    try:
        top_crop = randint(35, int(height / 8))
        bottom_crop = height - randint(35, int(height / 8))
        left_crop = randint(35, int(width / 8))
        right_crop = width - randint(35, int(width / 8))
    except ValueError:
        top_crop = randint(0, int(height / 8))
        bottom_crop = height - randint(0, int(height / 8))
        left_crop = randint(0, int(width / 8))
        right_crop = width - randint(0, int(width / 8))
    crop_img = img[top_crop:bottom_crop, left_crop:right_crop]
    return crop_img


def blur(img, kernel_size):
    blur_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blur_img


def random_rotate(img):
    negative_angle = randint(-20, -10)
    positive_angle = randint(10, 20)
    if abs(negative_angle) >= positive_angle:
        angle = negative_angle
    else:
        angle = positive_angle
    height = img.shape[0]
    width = img.shape[1]
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_img = cv2.warpAffine(img, M, (width, height))
    return rotated_img


def random_rotate_zoom(img):
    negative_angle = randint(-20, -10)
    positive_angle = randint(10, 20)
    if abs(negative_angle) >= positive_angle:
        angle = negative_angle
    else:
        angle = positive_angle
    zoom = uniform(1.1, 1.4)
    height = img.shape[0]
    width = img.shape[1]
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.2)
    rotated_zoomed_img = cv2.warpAffine(img, M, (width, height))
    return rotated_zoomed_img


def random_translate(img):
    height, width, channels = img.shape
    vertical_translation = randint(- int(height / 25), int(height / 25))
    horizontal_translation = randint(- int(width / 25), int(width / 25))
    M = np.float32([[1, 0, horizontal_translation],
                    [0, 1, vertical_translation]])
    translated_img = cv2.warpAffine(img, M, (width, height))
    # rotated_translated_img = random_rotate(dst)
    return translated_img


def jittering(img):
    height, width, channels = img.shape
    noise = np.random.randint(0, 35, (height, width))
    zitter = np.zeros_like(img)
    zitter[:, :, 1] = noise
    jittered_img = cv2.add(img, zitter)
    return jittered_img
