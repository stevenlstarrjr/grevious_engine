"""CPU probe for the procedural cloud field used by genesis_atmosphere.glsl.

This is not a renderer replacement. It cheaply ranks candidate weather offsets
before the Vulkan screenshot sweep verifies the actual image.
"""

from __future__ import annotations

import argparse
import math


def fract(value: float) -> float:
    return value - math.floor(value)


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def mul(a: tuple[float, float, float], scalar: float) -> tuple[float, float, float]:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def cloud_gradient(cell: tuple[float, float, float]) -> tuple[float, float, float]:
    cell = (
        fract(cell[0] * 0.1031),
        fract(cell[1] * 0.1030),
        fract(cell[2] * 0.0973),
    )
    scalar = dot(cell, (cell[1] + 33.33, cell[0] + 33.33, cell[2] + 33.33))
    cell = add(cell, (scalar, scalar, scalar))
    gradient = (
        fract((cell[0] + cell[1]) * cell[2]) * 2.0 - 1.0,
        fract((cell[0] + cell[0]) * cell[1]) * 2.0 - 1.0,
        fract((cell[1] + cell[0]) * cell[0]) * 2.0 - 1.0,
    )
    inverse_length = 1.0 / math.sqrt(max(dot(gradient, gradient), 1.0e-4))
    return mul(gradient, inverse_length)


def cloud_noise(point: tuple[float, float, float]) -> float:
    skew = 1.0 / 3.0
    unskew = 1.0 / 6.0
    skewed = dot(point, (skew, skew, skew))
    cell = tuple(math.floor(component + skewed) for component in point)
    unskewed = dot(cell, (unskew, unskew, unskew))
    x0 = tuple(point[index] - cell[index] + unskewed for index in range(3))
    order = (
        1.0 if x0[0] >= x0[1] else 0.0,
        1.0 if x0[1] >= x0[2] else 0.0,
        1.0 if x0[2] >= x0[0] else 0.0,
    )
    inverse = tuple(1.0 - component for component in order)
    inverse_zxy = (inverse[2], inverse[0], inverse[1])
    i1 = tuple(min(order[index], inverse_zxy[index]) for index in range(3))
    i2 = tuple(max(order[index], inverse_zxy[index]) for index in range(3))
    x1 = tuple(x0[index] - i1[index] + unskew for index in range(3))
    x2 = tuple(x0[index] - i2[index] + 2.0 * unskew for index in range(3))
    x3 = tuple(x0[index] - 1.0 + 3.0 * unskew for index in range(3))
    corners = ((cell, x0), (add(cell, i1), x1), (add(cell, i2), x2),
               (add(cell, (1.0, 1.0, 1.0)), x3))
    contribution = 0.0
    for corner, relative in corners:
        attenuation = max(0.6 - dot(relative, relative), 0.0)
        attenuation *= attenuation
        attenuation *= attenuation
        contribution += attenuation * dot(cloud_gradient(corner), relative)
    return min(max(0.5 + 21.0 * contribution, 0.0), 1.0)


def rotate_linear(point: tuple[float, float, float]) -> tuple[float, float, float]:
    # GLSL mat3 constructors list columns, not rows.
    rotated = (
        -0.80 * point[1] - 0.60 * point[2],
        0.80 * point[0] + 0.36 * point[1] - 0.48 * point[2],
        0.60 * point[0] - 0.48 * point[1] + 0.64 * point[2],
    )
    return rotated


def rotate_octave(point: tuple[float, float, float]) -> tuple[float, float, float]:
    return add(mul(rotate_linear(point), 2.03), (17.1, 9.2, 13.7))


def cloud_fbm(point: tuple[float, float, float], detail: float = 1.0) -> float:
    value = 0.5 + (cloud_noise(point) - 0.5) * 0.62
    point = rotate_octave(point)
    value += (cloud_noise(point) - 0.5) * 0.25 * (0.35 + 0.65 * detail)
    point = rotate_octave(point)
    value += (cloud_noise(point) - 0.5) * 0.13 * detail
    return min(max(value, 0.0), 1.0)


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    amount = min(max((value - edge0) / (edge1 - edge0), 0.0), 1.0)
    return amount * amount * (3.0 - 2.0 * amount)


def column_density(offset_x: float, offset_y: float, coverage: float = 0.42,
                   sample_x: float | None = None,
                   sample_y: float = 0.0) -> float:
    camera_x = math.sin(1.47) * 9.0 if sample_x is None else sample_x
    scale = 12.0
    horizontal = ((camera_x - offset_x) / scale,
                  (sample_y - offset_y) / scale)
    weather_coordinate = (horizontal[0] * 0.12, horizontal[1] * 0.12, 3.7)
    weather_warp = (
        cloud_noise(add(weather_coordinate, (2.1, 7.7, 4.2))) - 0.5,
        cloud_noise(add(weather_coordinate, (8.4, 1.3, 6.5))) - 0.5,
    )
    warped = (horizontal[0] + weather_warp[0] * 2.0,
              horizontal[1] + weather_warp[1] * 2.0)
    weather = cloud_noise((warped[0] * 0.22 + 4.7,
                           warped[1] * 0.22 + 11.3, 2.1))
    footprint = cloud_fbm((warped[0] * 2.10 + 7.9,
                           warped[1] * 2.10 + 2.6, 5.8), 1.0)
    base_variation = cloud_noise((warped[0] * 3.40 + 3.4,
                                  warped[1] * 3.40 + 14.8, 9.6))
    column_field = (weather * 0.44 + footprint * 0.56 +
                    (base_variation - 0.5) * 0.12)
    threshold = 0.76 + (0.27 - 0.76) * coverage
    column_mask = smoothstep(threshold - 0.025, threshold + 0.040,
                             column_field)
    total = 0.0
    samples = 17
    for index in range(1, samples + 1):
        height = index / (samples + 1.0)
        shape_coordinate = (warped[0] * 4.2 + 19.1,
                            warped[1] * 4.2 + 3.7,
                            height * 3.6 + 7.4)
        shape = cloud_fbm(shape_coordinate)
        erosion_point = add(mul(rotate_linear(shape_coordinate), 2.2),
                            (1.2, 8.4, 15.7))
        erosion = cloud_noise(erosion_point)
        sample_column_field = (weather * 0.44 + footprint * 0.56 +
                               (base_variation - 0.5) * 0.10 +
                               (erosion - 0.5) * 0.16)
        column_mask = smoothstep(threshold - 0.025, threshold + 0.040,
                                 sample_column_field)
        billow = smoothstep(0.24, 0.70, shape)
        formed = column_mask * (0.34 + 0.66 * billow)
        growth = column_mask * smoothstep(
            0.28, 0.78, sample_column_field * 0.70 + shape * 0.30)
        cloud_top = 0.46 + (0.97 - 0.46) * growth
        cloud_base = 0.01 + 0.12 * base_variation
        lower = smoothstep(cloud_base, cloud_base + 0.10, height)
        upper = 1.0 - smoothstep(max(cloud_top - 0.20, 0.12), cloud_top, height)
        body = 0.42 + (1.0 - 0.42) * smoothstep(0.20, 0.82, shape)
        edge = column_mask * (1.0 - smoothstep(0.35, 0.82, billow))
        formed = min(max(formed - (1.0 - erosion) * 0.15 * edge, 0.0), 1.0)
        total += formed * lower * upper * body
    return total / samples


def patch_statistics(offset_x: float, offset_y: float, coverage: float,
                     radius: float, sample_x: float | None = None,
                     sample_y: float = 0.0) -> tuple[float, float, float, float]:
    center_x = math.sin(1.47) * 9.0 if sample_x is None else sample_x
    samples: list[float] = []
    for y_step in range(-2, 3):
        for x_step in range(-2, 3):
            samples.append(column_density(
                offset_x, offset_y, coverage,
                center_x + radius * x_step / 2.0,
                sample_y + radius * y_step / 2.0))
    mean = sum(samples) / len(samples)
    variance = sum((value - mean) ** 2 for value in samples) / len(samples)
    return mean, math.sqrt(variance), min(samples), max(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--center-x', type=float, default=48.0)
    parser.add_argument('--center-y', type=float, default=-35.5)
    parser.add_argument('--radius', type=float, default=72.0)
    parser.add_argument('--step', type=float, default=4.0)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--coverage', type=float, default=0.42)
    parser.add_argument('--target', type=float, default=0.18)
    parser.add_argument('--patch-radius', type=float, default=4.0)
    parser.add_argument('--sample-x', type=float)
    parser.add_argument('--sample-y', type=float, default=0.0)
    arguments = parser.parse_args()
    candidates: list[tuple[float, float, float, float, float, float]] = []
    steps = int(arguments.radius / arguments.step)
    for y_index in range(-steps, steps + 1):
        for x_index in range(-steps, steps + 1):
            offset_x = arguments.center_x + x_index * arguments.step
            offset_y = arguments.center_y + y_index * arguments.step
            mean, deviation, minimum, maximum = patch_statistics(
                offset_x, offset_y, arguments.coverage, arguments.patch_radius,
                arguments.sample_x, arguments.sample_y)
            candidates.append((mean, deviation, minimum, maximum,
                               offset_x, offset_y))
    # Prefer partially cloudy patches with visible local contrast. Very high
    # variance is useful here: it puts both cloud and blue sky in an upward FOV.
    candidates.sort(key=lambda item: abs(item[0] - arguments.target) -
                     min(item[1], 0.16) * 0.9)
    print(' mean    sigma     min     max    offset_x  offset_y')
    for mean, deviation, minimum, maximum, offset_x, offset_y in candidates[:arguments.count]:
        print(f'{mean:7.4f}  {deviation:7.4f}  {minimum:6.3f}  {maximum:6.3f}'
              f'  {offset_x:8.1f}  {offset_y:8.1f}')


if __name__ == '__main__':
    main()
