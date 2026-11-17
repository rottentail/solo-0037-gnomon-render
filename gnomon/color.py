import numpy as np


def srgb_to_linear(value):
    value = np.asarray(value, dtype=np.float64)
    return np.where(value <= 0.04045, value/12.92, ((value+0.055)/1.055)**2.4)


def linear_to_srgb(value):
    value = np.clip(value, 0, 1)
    return np.where(value <= 0.0031308, value*12.92, 1.055*value**(1/2.4)-0.055)


def encode_rgb(value):
    return np.rint(linear_to_srgb(value)*255).astype(np.uint8)
