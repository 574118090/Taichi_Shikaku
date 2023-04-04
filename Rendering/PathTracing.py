import taichi as ti
import taichi.math as tm

import Math
import SDF

from Config import SAMPLE_PER_PIXELS, QUALITY_PER_SAMPLE, SCREEN_PIXEL_SIZE, ASPECT_RATIO, TMAX
from Config import VISIBILITY, PIXEL_RADIUS, APERTURE, FOV, FOCUS, IOR, TMIN, MAX_RAYMARCH
from Camera import generate_ray

from Ray import Ray

from Math import hemispheric_sampling, hemispheric_sampling_roughness, hemispheric_sampling_light

from Scene import scene_objects, background_shading, lights_num

from Fields import image_pixels, image_buffer, ray_buffer


@ti.func
def ray_cast(ray) -> tuple[Ray, int, bool]:
    t, w, s, distance = 0.0, 0.6, 0.0, TMAX
    index, hit = -1, False
    for i in range(MAX_RAYMARCH):
        ld = distance
        index, distance = SDF.nearest_SDFobject(ray.origin)

        if w > 1.0 and ld + distance < s:
            s -= w * s
            w = 1.0
            t += s
            ray.origin += ray.direction * s
            continue
        s = w * distance
        t += s
        ray.origin += ray.direction * s

        hit = distance < t * PIXEL_RADIUS
        if hit or t >= TMAX:
            break

    ray.depth += 1
    return ray, index, hit


@ti.func
def BSDF(p: tm.vec3, l: tm.vec3, u: tm.vec3, ray: Ray, i: int, j: int) -> Ray:
    xi = 1.0 if ray.depth == 0 else QUALITY_PER_SAMPLE
    xi -= ray.depth / MAX_RAYMARCH

    # russian_roulette
    save_ray = ray
    ray.color = tm.vec4(0)
    ray.depth *= -1

    pdf = ti.random()
    if pdf <= xi:
        ray = save_ray
        ray.color *= 1.0 / xi
        if ray.depth < 1:
            image_buffer[i, j] += tm.vec4(ray.color.xyz, 1.0)

            coord = tm.vec2(i, j) + tm.vec2(ti.random(), ti.random())
            uv = coord * SCREEN_PIXEL_SIZE
            ray = generate_ray(p, l, u, uv)

        ray, index, hit = ray_cast(ray)

        # ray cast
        if hit:
            albedo = scene_objects[index].mtl.albedo
            roughness = scene_objects[index].mtl.roughness
            metallic = scene_objects[index].mtl.metallic
            transmission = scene_objects[index].mtl.transmission
            ior = scene_objects[index].mtl.ior
            emission_color = scene_objects[index].mtl.emission_color
            emission_intensity = scene_objects[index].mtl.emission_intensity

            normal = SDF.get_SDF_normal(scene_objects[index], ray.origin)
            outer = tm.dot(ray.direction, normal) < 0.0
            normal *= 1.0 if outer else -1.0

            alpha = roughness * roughness
            hemispheric_sample = hemispheric_sampling(normal)
            roughness_sample = tm.normalize(tm.mix(normal, hemispheric_sample, alpha))

            N = roughness_sample
            I = ray.direction
            NoI = tm.dot(N, I)

            eta = IOR / ior if outer else ior / IOR
            k = 1.0 - eta * eta * (1.0 - NoI * NoI)
            F0 = 2.0 * (eta - 1.0) / (eta + 1.0)
            F = Math.fresnel_schlick(NoI, F0 * F0)

            contribution = albedo
            contribution_offset = 1.0

            specular_reflect_pdf = ti.random()
            transmission_pdf = ti.random()
            light_pdf = ti.random()

            if light_pdf > xi:
                ray.direction = hemispheric_sampling_light(ray.origin)
                outer = tm.dot(ray.direction, normal) < 0.0
                ray.direction = hemispheric_sample if outer else ray.direction
            elif specular_reflect_pdf < F + metallic or k < 0.0:
                ray.direction = I - 2.0 * NoI * N
                outer = tm.dot(ray.direction, normal) < 0.0
                ray.direction *= (-1.0 if outer else 1.0)
            elif transmission_pdf < transmission:
                ray.direction = eta * I - (tm.sqrt(abs(k)) + eta * NoI) * N
            else:
                ray.direction = hemispheric_sample

            outer = tm.dot(ray.direction, normal) < 0.0
            ray.color *= tm.vec4(albedo, 1)
            ray.origin += normal * TMIN * (-1.0 if outer else 1.0)

            if emission_intensity > 0:
                ray.color.xyz *= emission_color * emission_intensity
                # contribution *= emission_color
                # contribution_offset *= emission_intensity
                ray.depth *= -1

            # ray.color *= contribution * contribution_offset

        else:
            ray.depth *= -1
            ray.color.xyz *= background_shading()

    ray.direction = tm.normalize(ray.direction)
    return ray


@ti.func
def sample(p: tm.vec3, l: tm.vec3, u: tm.vec3, i: int, j: int):
    ray = ray_buffer[i, j]

    for _ in range(SAMPLE_PER_PIXELS):
        ray = BSDF(p, l, u, ray, i, j)

    ray_buffer[i, j] = ray


@ti.kernel
def path_trace(p: tm.vec3, l: tm.vec3, u: tm.vec3):
    for i, j in image_pixels:
        sample(p, l, u, i, j)


@ti.kernel
def refresh():
    for i, j in image_pixels:
        image_pixels[i, j] = ti.math.vec3(0)
        image_buffer[i, j] *= 0.33
        ray_buffer[i, j].depth = -1
