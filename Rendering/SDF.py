import taichi as ti
import taichi.math as tm

from Object import SDFObject, BoundingBox
from Object import CUBE, SPHERE, TRIANGULAR_PRISM

from ObjectData import scene_objects, object_num, scene_octree

from Ray import Ray


@ti.func
def sdf_sphere(p: tm.vec3, r: float) -> float:
    return tm.length(p) - r


@ti.func
def sdf_cube(p: tm.vec3, b: tm.vec3) -> float:
    q = abs(p) - b
    return tm.length(tm.max(q, 0)) + tm.min(tm.max(q.x, q.y, q.z), 0)


@ti.func
def sdf_triangular_prism(p: tm.vec3, s: tm.vec3) -> float:
    q = abs(p)
    return tm.max(q.z - s.y, tm.max(q.x * 0.866025 + p.y * 0.5, -p.y) - s.x * 0.5)


@ti.func
def signed_SDF_distance(obj: SDFObject, pos: tm.vec3) -> float:
    position = obj.trans.position
    scale = obj.trans.scale

    p = position - pos
    p = obj.trans.rotation_matrix @ p
    sd = 0.0
    if obj.type == SPHERE:
        sd = sdf_sphere(p, scale.x)
    elif obj.type == CUBE:
        sd = sdf_cube(p, scale)
    elif obj.type == TRIANGULAR_PRISM:
        sd = sdf_triangular_prism(p, scale)
    else:
        sd = sdf_sphere(p, scale.x)
    return sd


@ti.func
def get_SDF_normal(obj, p: tm.vec3) -> tm.vec3:
    e = tm.vec2(1, -1) * 0.5773 * 0.0005
    return tm.normalize(e.xyy * signed_SDF_distance(obj, p + e.xyy) +
                        e.yyx * signed_SDF_distance(obj, p + e.yyx) +
                        e.yxy * signed_SDF_distance(obj, p + e.yxy) +
                        e.xxx * signed_SDF_distance(obj, p + e.xxx))


@ti.func
def nearest_SDFobject(p: tm.vec3) -> tuple[int, float]:
    index = -1
    res = 1e9
    for i in range(object_num):
        dis = abs(signed_SDF_distance(scene_objects[i], p))
        if dis < res:
            index = i
            res = dis
    return index, res


@ti.func
def in_box_ray(self: BoundingBox, p: tm.vec3, d: tm.vec3):
    t_min_x = (self.p_min.x - p.x) / d.x
    t_max_x = (self.p_max.x - p.x) / d.x

    t_min = tm.min(t_min_x, t_max_x)
    t_max = tm.max(t_min_x, t_max_x)

    y_min = p.y + d.y * t_min
    y_max = p.y + d.y * t_max

    z_min = p.z + d.z * t_min
    z_max = p.z + d.z * t_max

    if y_min > y_max:
        y_min, y_max = y_max, y_min
    if z_min > z_max:
        z_min, z_max = z_max, z_min

    a = (y_min < self.p_max.y) and (y_max > self.p_min.y)
    b = (z_min < self.p_max.z) and (z_max > self.p_min.z)

    return a and b


def get_SDF_bounding_box(obj: SDFObject):
    if obj.type == SPHERE:
        return BoundingBox(obj.trans.position + tm.vec3(-obj.trans.scale.x),
                           obj.trans.position + tm.vec3(obj.trans.scale.x), 0)
    if obj.type == CUBE:
        return BoundingBox(obj.trans.position - obj.trans.scale, obj.trans.position + obj.trans.scale, 0)

    else:
        return BoundingBox(obj.trans.position - obj.trans.scale, obj.trans.position + obj.trans.scale, 0)
