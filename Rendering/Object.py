import taichi as ti
import taichi.math as tm

NONE = 0
SPHERE = 1
CUBE = 2
TRIANGULAR_PRISM = 3


@ti.dataclass
class Material:
    albedo: tm.vec3  # 固有色
    emission_color: tm.vec3  # 自发光
    emission_intensity: float
    roughness: float  # 粗糙度
    metallic: float  # 金属度
    transmission: float  # 透明度
    ior: float  # 折射率

    def __init__(self, albedo, emission_color, emission_intensity, roughness, metallic, transmission, ior):
        self.albedo = albedo
        self.roughness = roughness
        self.metallic = metallic
        self.transmission = transmission
        self.ior = ior
        self.emission_color = emission_color
        self.emission_intensity = emission_intensity


@ti.dataclass
class Transform:
    position: tm.vec3
    scale: tm.vec3
    rotation: tm.vec3
    rotation_matrix: tm.mat3
    scale_matrix: tm.mat3
    trans_matrix: tm.mat3

    def __init__(self, pos: tm.vec3, scl: tm.vec3, rot: tm.vec3):
        self.position = pos
        self.scale = scl
        self.rotation = rot

    def update_matrix(self):
        s = tm.sin(self.rotation)
        c = tm.cos(self.rotation)
        self.rotation_matrix = tm.mat3(tm.vec3(c.z, s.z, 0),
                                       tm.vec3(-s.z, c.z, 0),
                                       tm.vec3(0, 0, 1)) @ \
                               tm.mat3(tm.vec3(c.y, 0, -s.y),
                                       tm.vec3(0, 1, 0),
                                       tm.vec3(s.y, 0, c.y)) @ \
                               tm.mat3(tm.vec3(1, 0, 0),
                                       tm.vec3(0, c.x, s.x),
                                       tm.vec3(0, -s.x, c.x))
        self.scale_matrix = tm.mat3(tm.vec3(self.scale.x, 0, 0),
                                    tm.vec3(0, self.scale.y, 0),
                                    tm.vec3(0, 0, self.scale.z)
                                    )


@ti.dataclass
class BoundingBox:
    p_min: tm.vec3
    p_max: tm.vec3
    layer: int
    l_flag: int
    r_flag: int
    obj_id: int

    # vertices: ti.Vector.field(3, float)

    def __init__(self, i: tm.vec3, a: tm.vec3, l: int):
        self.p_min = i
        self.p_max = a
        self.layer = l
        self.l_flag = 0
        self.r_flag = 0
        # ti.root.dense(ti.i, 8).place(self.vertices)

    def corner(self, index) -> tm.vec3:
        a = self.p_min.x if bool(index & 1) else self.p_max.x
        b = self.p_min.y if bool(index & 2) else self.p_max.y
        c = self.p_min.z if bool(index & 4) else self.p_max.z
        return tm.vec3(a, b, c)

    @ti.func
    def in_box_ndc(self, p: tm.vec2) -> bool:
        return (self.p_min.x <= p.x <= self.p_max.x) and \
               (self.p_min.y <= p.y <= self.p_max.y)


@ti.dataclass
class SDFObject:
    type: ti.u32
    mtl: Material
    trans: Transform
    sd: float

    def __init__(self, type: ti.u32, trans: Transform, mtl: Material):
        self.type = type
        self.trans = trans
        self.mtl = mtl
        self.sd = 1e9


@ti.func
def inter_bounding(p: tm.vec3, b: BoundingBox) -> bool:
    return (b.p_min.x < p < b.p_max.x) and (b.p_min.y < p < b.p_max.y) and (b.p_min.z < p < b.p_max.z)


@ti.func
def bounding_distance(p: tm.vec3, b: BoundingBox) -> float:
    scale = (b.p_max - b.p_min) / 2.0
    pos = (b.p_max + b.p_min) / 2.0
    position = p - pos
    q = abs(position) - scale
    return tm.length(tm.max(q, 0)) + tm.min(tm.max(q.x, q.y, q.z), 0)


def merge_bounding(a: BoundingBox, b: BoundingBox):
    a_min = a.p_min
    a_max = a.p_max
    b_min = b.p_min
    b_max = b.p_max

    m_min = tm.vec3(min(a_min.x, b_min.x), min(a_min.y, b_min.y), min(a_min.z, b_min.z))
    m_max = tm.vec3(max(a_max.x, b_max.x), max(a_max.y, b_max.y), max(a_max.z, b_max.z))

    return BoundingBox(m_min, m_max, max(a.layer, b.layer) + 1)
