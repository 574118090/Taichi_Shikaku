import taichi as ti

from taichi.math import vec2, vec3, vec4
from taichi.math import radians, normalize, cross, tan

from Ray import Ray

from Math import random_in_unit_disk

from Config import ASPECT_RATIO, FOV, APERTURE, FOCUS


@ti.dataclass
class Camera:
    position: vec3
    lookat: vec3
    up: vec3
    fov: float  # 纵向视野
    aspect: float  # 长宽比
    aperture: float  # 光圈大小
    focus: float  # 焦距


@ti.func
def get_ray(c, uv: vec2, color: vec4) -> Ray:
    theta = radians(c.fov)
    half_height = tan(theta * 0.5)
    half_width = c.aspect * half_height

    z = normalize(c.position - c.lookat)
    x = normalize(cross(c.up, z))
    y = cross(z, x)

    start_pos = c.position - half_width * c.focus * x - half_height * c.focus * y - c.focus * z

    horizontal = 2.0 * half_width * c.focus * x
    vertical = 2.0 * half_height * c.focus * y

    lens_radius = c.aperture * 0.5  # 透镜半径 = 光圈大小的一半（归一化）
    rud = lens_radius * random_in_unit_disk()  # 随机采样点 = 光圈范围 * 随机采样
    offset = x * rud.x + y * rud.y  # 计算光圈中随机采样点的偏移量

    # 计算光线
    ro = c.position + offset
    rp = start_pos + uv.x * horizontal + uv.y * vertical
    rd = normalize(rp - ro)

    return Ray(ro, rd, color)


@ti.func
def generate_ray(position: vec3, lookat: vec3, up: vec3, uv: vec2) -> Ray:
    camera = Camera()
    camera.position = position
    camera.lookat = lookat
    camera.up = up
    camera.aspect = ASPECT_RATIO
    camera.fov = FOV
    camera.aperture = APERTURE
    camera.focus = FOCUS
    return get_ray(camera, uv, vec4(1))


@ti.func
def generate_ray_direct(position: vec3, lookat: vec3, up: vec3, uv: vec2) -> Ray:
    theta = radians(FOV)
    half_height = tan(theta * 0.5)
    half_width = ASPECT_RATIO * half_height

    z = normalize(position - lookat)
    x = normalize(cross(up, z))
    y = cross(z, x)

    horizontal = 2.0 * half_width * FOCUS * x
    vertical = 2.0 * half_height * FOCUS * y

    start_pos = position - half_width * FOCUS * x - half_height * FOCUS * y - FOCUS * z
    ro = position
    rp = start_pos + uv.x * horizontal + uv.y * vertical
    rd = normalize(rp - ro)

    return Ray(ro, rd, vec4(0))
