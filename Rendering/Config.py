import taichi as ti
import taichi.math as tm

# renderer scope
RESOLUTION_RATIO = (1024, 1024)  # 分辨率
ASPECT_RATIO = RESOLUTION_RATIO[0] / RESOLUTION_RATIO[1]  # 纵横比
SCREEN_PIXEL_SIZE = 1.0 / tm.vec2(RESOLUTION_RATIO)  # 像素大小

PIXEL_RADIUS = 1.0 * SCREEN_PIXEL_SIZE.min()  # 像素半径（按较小值计算）

VISIBILITY = tm.vec2(1e-4, 1e3)  # 可见范围

QUALITY_PER_SAMPLE = 0.9  # 每像素渲染质量
SAMPLE_PER_PIXELS = 1  # 每像素采样次数

# scene scope
scene_min_position = tm.vec3(0)
scene_max_position = tm.vec3(0)
SPACE_SUBDIVISION = 850

# camera scope
FOV = 30
APERTURE = 0.01
FOCUS = 4
MOVING_SPEED = 0.03

taichi_camera = ti.ui.Camera()

# ray-cast scope
TMIN = 2.5 * PIXEL_RADIUS
TMAX = 1e3
PRECISION = 1e-4  # @Shadow Acne

MAX_RAYMARCH = 512
MAX_PATHTRACE = 512

IOR = 1.000277

# light scope
LIGHT_NUM = 0
# color scope
