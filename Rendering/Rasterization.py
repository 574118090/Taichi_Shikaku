import taichi as ti

from taichi.math import vec2, vec3, vec4

from Config import RESOLUTION_RATIO
from Config import taichi_camera, SCREEN_PIXEL_SIZE, scene_max_position, scene_min_position

from Fields import image_buffer

from Camera import generate_ray, generate_ray_direct

from PathTracing import ray_cast

from ObjectData import scene_octree, scene_objects, object_num

from SDF import in_box_ray, signed_SDF_distance

from Ray import Ray

d_buffer = ti.field(float)
ti.root.dense(ti.ij, RESOLUTION_RATIO).place(d_buffer)

shade_buffer = ti.Vector.field(3, float)
ti.root.dense(ti.ij, RESOLUTION_RATIO).place(shade_buffer)

box = (scene_max_position - scene_min_position)


def rasterization(p: vec3, l: vec3, u: vec3):
    depth_test(p, l, u)
    shade(p, l, u)
    draw()


@ti.func
def ray_cast_direct(ray: Ray) -> bool:
    hit = False
    for i in range(object_num):
        if in_box_ray(scene_octree[i], ray.origin, ray.direction):
            hit = True

    return hit


@ti.kernel
def draw():
    for i, j in image_buffer:
        # 深度图
        # image_buffer[i, j] = vec4(d_buffer[i, j], d_buffer[i, j], d_buffer[i, j], 1)

        # 着色图
        image_buffer[i, j] = vec4(shade_buffer[i, j], 1)


@ti.kernel
def depth_test(p: vec3, l: vec3, u: vec3):
    for i, j in d_buffer:
        z = d_buffer[i, j]

        coord = vec2(i, j)
        uv = coord * SCREEN_PIXEL_SIZE

        d_buffer[i, j] = ergodic_depth(p, l, u, uv)


@ti.func
def ergodic_depth(p: vec3, l: vec3, u: vec3, uv: vec2) -> float:
    ray = generate_ray_direct(p, l, u, uv)
    ray, index, hit = ray_cast(ray)
    return (ray.origin.z + box.z) / box.z


@ti.kernel
def shade(p: vec3, l: vec3, u: vec3):
    for i, j in shade_buffer:
        coord = vec2(i, j)
        uv = coord * SCREEN_PIXEL_SIZE

        ray = generate_ray_direct(p, l, u, uv)
        index = -1
        hit = False
        if ray_cast_direct(ray):
            ray, index, hit = ray_cast(ray)
        if hit:
            shade_buffer[i, j] = scene_objects[index].mtl.albedo
        else:
            shade_buffer[i, j] = vec3(0)
