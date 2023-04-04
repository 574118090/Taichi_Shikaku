import taichi as ti
import time

import Initial
import PathTracing
import Rasterization

from Config import RESOLUTION_RATIO, MOVING_SPEED
from Config import taichi_camera

from Fields import image_pixels, image_buffer, ray_buffer

from PostProcess import post_process

refresh_frame = True


def render():
    if refresh_frame:
        PathTracing.refresh()
    else:
        # Rasterization.rasterization(taichi_camera.curr_position, taichi_camera.curr_lookat, taichi_camera.curr_up)
        PathTracing.path_trace(taichi_camera.curr_position, taichi_camera.curr_lookat, taichi_camera.curr_up)
    post_process()


@ti.kernel
def clear():
    for i, j in image_pixels:
        image_pixels[i, j] = ti.math.vec3(0)
        image_buffer[i, j] *= 0
        ray_buffer[i, j].depth = -1


window = ti.ui.Window("Shikaku Renderer", RESOLUTION_RATIO)
canvas = window.get_canvas()

refresh = False

taichi_camera.position(0, 0, 4.5)
taichi_camera.lookat(0, 0, 1)
taichi_camera.up(0, 1, 0)

while window.running:
    start_time = time.time()
    taichi_camera.track_user_inputs(window, movement_speed=MOVING_SPEED, hold_key=ti.ui.LMB)

    dt = time.time() - start_time

    # refresh camera
    refresh_frame = False
    # moving
    x = int(window.is_pressed('d')) - int(window.is_pressed('a'))
    y = int(window.is_pressed('w')) - int(window.is_pressed('s'))

    if window.is_pressed(ti.ui.LMB) or x != 0 or y != 0:
        clear()

    render()
    canvas.set_image(image_pixels)

    window.show()
