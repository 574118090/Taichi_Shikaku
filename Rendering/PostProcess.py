import taichi as ti
import taichi.math as tm
import numpy as np

from Fields import image_pixels, image_buffer

from Config import RESOLUTION_RATIO


def post_process():
    load()
    # bloom()


@ti.kernel
def load():
    for i, j in image_pixels:
        buffer = image_buffer[i, j]

        gamma = ti.static(1.0 / 2.4)

        color = buffer.rgb / buffer.a
        color = tm.pow(color, gamma)

        image_pixels[i, j] = color

        image_pixels[i, j] = tm.clamp(image_pixels[i, j], 0, 1)
