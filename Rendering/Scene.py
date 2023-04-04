import taichi as ti

import SDF
import Config

from taichi.math import floor
from math import ceil, log2

from Object import BoundingBox
from Object import merge_bounding

from ObjectData import *


@ti.func
def background_shading() -> vec3:
    return vec3(0)


def update_transform():
    for i in range(object_num):
        scene_objects[i].trans.update_matrix()

        scene_octree[i] = SDF.get_SDF_bounding_box(scene_objects[i])
        scene_octree[i].obj_id = i

        Config.scene_min_position = vec3(min(Config.scene_min_position.x, scene_octree[i].p_min.x),
                                         min(Config.scene_min_position.y, scene_octree[i].p_min.y),
                                         min(Config.scene_min_position.z, scene_octree[i].p_min.z))

        Config.scene_max_position = vec3(max(Config.scene_max_position.x, scene_octree[i].p_min.x),
                                         max(Config.scene_max_position.y, scene_octree[i].p_min.y),
                                         max(Config.scene_max_position.z, scene_octree[i].p_min.z))

        if scene_objects[i].mtl.emission_intensity > 0:
            Config.LIGHT_NUM += 1
    print("Info:Light num is:", Config.LIGHT_NUM)


def build_octree() -> float:
    aim_layer = label_num
    for i in range(object_num):
        scene_octree[i] = SDF.get_SDF_bounding_box(scene_objects[i])
    flag = 0
    while True:
        new_flag = object_num + flag // 2
        scene_octree[new_flag] = merge_bounding(scene_octree[flag], scene_octree[flag + 1])
        scene_octree[new_flag].l_flag = flag
        scene_octree[new_flag].r_flag = flag + 1
        if scene_octree[new_flag].layer == aim_layer:
            break
        flag += 2
    return object_num + flag // 2


@ti.kernel
def build_subdivision():
    for i, j, k in space_subdivision:
        offset = 0.5 - Config.SPACE_SUBDIVISION // 2
        index, res = SDF.nearest_SDFobject(
            vec3((i + offset) * per_subdivision, (j + offset) * per_subdivision, (k + offset) * per_subdivision))
        space_subdivision[i, j, k] = index


# 还不成熟的技术，待更新
@ti.func
def query_subdivision(p: vec3) -> tuple[int, float]:
    offset = Config.SPACE_SUBDIVISION // 2
    i = int(floor(p.x / per_subdivision) + offset)
    j = int(floor(p.y / per_subdivision) + offset)
    k = int(floor(p.z / per_subdivision) + offset)
    index = space_subdivision[i, j, k]
    ris = abs(SDF.signed_SDF_distance(scene_objects[index], p))
    return index, ris


lights_index = 0

for i in range(len(objects)):
    scene_objects[i] = objects[i]
    if objects[i].mtl.emission_intensity > 0:
        if lights_index == 0:
            lights_index = i

lights_num = object_num - lights_index

label_num = ceil(log2(object_num))

space_subdivision = ti.field(
    int,
    shape=(Config.SPACE_SUBDIVISION, Config.SPACE_SUBDIVISION, Config.SPACE_SUBDIVISION)
)

update_transform()
octree_num = build_octree() if object_num > 1 else 1
per_subdivision = 100 / Config.SPACE_SUBDIVISION
build_subdivision()

print("Info:The Range of the Scene:", Config.scene_min_position, " —— ", Config.scene_max_position)
