import taichi as ti
import taichi.math as tm


@ti.dataclass
class Ray:
    origin: tm.vec3
    direction: tm.vec3
    color: tm.vec4
    depth: int
