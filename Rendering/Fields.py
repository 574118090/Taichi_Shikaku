import taichi as ti

from Config import RESOLUTION_RATIO
from Ray import Ray

# field scope
ray_buffer = Ray.field()
image_buffer = ti.Vector.field(4, float)
image_pixels = ti.Vector.field(3, float)

ti.root.dense(ti.ij, RESOLUTION_RATIO).place(ray_buffer)
ti.root.dense(ti.ij, RESOLUTION_RATIO).place(image_buffer)
ti.root.dense(ti.ij, RESOLUTION_RATIO).place(image_pixels)
