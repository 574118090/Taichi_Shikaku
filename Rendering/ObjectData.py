import taichi as ti

from taichi.math import vec3

from MaterialData import *

from SDF import SDFObject
from SDF import CUBE, SPHERE

from Object import Transform, BoundingBox

objects = sorted([
    SDFObject(type=CUBE,
              trans=Transform(vec3(-0.3, -0.34, -0.25), vec3(0.3, 0.6, 0.2), vec3(0, 65 * 3.14 / 360, 0)),
              mtl=white_iron
              ),
    SDFObject(type=CUBE,
              trans=Transform(vec3(0.35, -0.65, 0.2), vec3(0.3, 0.3, 0.3), vec3(0, -40 * 3.14 / 360, 0)),
              mtl=white_iron
              ),
    SDFObject(type=CUBE,
              trans=Transform(vec3(1.05, 0, 0), vec3(0.1, 1, 1), vec3(0, 0, 0)),
              mtl=green_plastic
              ),
    SDFObject(type=CUBE,
              trans=Transform(vec3(-1.05, 0, 0), vec3(0.1, 1, 1), vec3(0, 0, 0)),
              mtl=red_plastic
              ),
    SDFObject(type=CUBE,
              trans=Transform(vec3(0, 1.05, 0.1), vec3(0.9, .1, .9), vec3(0, 0, 0)),
              mtl=white_plastic
              ),
    SDFObject(type=CUBE,
              trans=Transform(vec3(0, -1.05, 0), vec3(1, .1, 1), vec3(0, 0, 0)),
              mtl=white_plastic
              ),
    SDFObject(type=CUBE,
              trans=Transform(vec3(0, 0, -1.05), vec3(1, 1, .1), vec3(0, 0, 0)),
              mtl=white_plastic
              ),
    SDFObject(type=CUBE,
              trans=Transform(vec3(0, 1.04, 0), vec3(0.2, .1, 0.2), vec3(0, 0, 0)),
              mtl=light
              )
], key=lambda o: o.mtl.emission_intensity)

scene_objects = SDFObject.field()
ti.root.dense(ti.i, len(objects)).place(scene_objects)

object_num = ti.static(len(objects))

scene_octree = BoundingBox.field(shape=object_num * 2)
