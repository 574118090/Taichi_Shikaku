from taichi.math import vec3
from Object import Material

# Material:
# albedo: vec3
# emission: tm.vec4  # 自发光
# roughness: float  # 粗糙度
# metallic: float  # 金属度
# transmission: float  # 透明度
# ior: float  # 折射率
white_plastic = Material(
    albedo=vec3(0.8, 0.8, 0.8) * 0.9,
    emission_intensity=0,
    roughness=1,
    metallic=0.8,
    transmission=0,
    ior=1
)
red_plastic = Material(
    albedo=vec3(0.6, 0.0, 0.0) * 0.9,
    emission_intensity=0,
    roughness=0.9,
    metallic=0.01,
    transmission=0,
    ior=1
)
green_plastic = Material(
    albedo=vec3(0.0, 0.6, 0.0) * 0.9,
    emission_intensity=0,
    roughness=0.9,
    metallic=0.01,
    transmission=0,
    ior=1
)
light = Material(
    albedo=vec3(1),
    emission_color=vec3(1, 0.4, 0.3),
    emission_intensity=5,
    roughness=1,
    metallic=1,
    transmission=0,
    ior=1)
glass = Material(
    albedo=vec3(1, 1, 1) * 0.99,
    emission_intensity=0,
    roughness=0,
    metallic=0,
    transmission=1,
    ior=1.88)
mirror = Material(
    albedo=vec3(1, 1, 1) * 0.9,
    emission_intensity=0,
    roughness=0,
    metallic=1,
    transmission=0,
    ior=1.03)
white_iron = Material(
    albedo=vec3(1, 1, 1) * 0.4,
    emission_intensity=0,
    roughness=0.88,
    metallic=0.4,
    transmission=0,
    ior=0.53)
