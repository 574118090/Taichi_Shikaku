import taichi as ti

from taichi.math import vec2, vec3
from taichi.math import dot, sin, cos, sqrt, mix, clamp, normalize
from taichi.math import pi

from Scene import lights_num, scene_objects, object_num, lights_index

from Ray import Ray


@ti.func
def brightnness(rgb: vec3) -> float:
    return dot(rgb, vec3(0.299, 0.587, 0.114))


@ti.func
def random_in_unit_sphere(offset: float) -> vec3:
    vec = vec2(ti.random(), ti.random())
    z = 2.0 * vec.x - 1.0
    a = vec.y * 2.0 * pi / offset

    xy = sqrt(1.0 - z * z) * vec2(sin(a), cos(a))
    return vec3(xy, z)


@ti.func
def random_in_unit_disk() -> vec2:
    x = ti.random()
    a = ti.random() * 2 * pi
    return sqrt(x) * vec2(sin(a), cos(a))


@ti.func
def fresnel_schlick(NoI: float, f0: float) -> float:
    return clamp(mix(pow(abs(1.0 + NoI), 5.0), 1.0, f0), 0, 1)


@ti.func
def hemispheric_sampling(n: vec3) -> vec3:
    vector = random_in_unit_sphere(1)
    return normalize(n + vector)


@ti.func
def hemispheric_sampling_light(p: vec3) -> vec3:
    a = (ti.random() - 0.5) * 2
    b = (ti.random() - 0.5) * 2
    c = (ti.random() - 0.5) * 2
    index = object_num - (ti.random(dtype=ti.i32) % lights_index + 1)
    pos = scene_objects[int(index)].trans.position + vec3(a, b, c) * scene_objects[int(index)].trans.scale
    l_r = pos - p
    l_r = normalize(l_r)
    return l_r


@ti.func
def hemispheric_sampling_roughness(hemisphere_sample: vec3, normal: vec3, roughness: float) -> vec3:
    alpha = roughness * roughness
    return normalize(mix(normal, hemisphere_sample, alpha))
