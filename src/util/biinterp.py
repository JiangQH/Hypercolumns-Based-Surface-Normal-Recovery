#!/usr/bin/env python
# coding=utf-8
"""
do bilinear interplation on the fly. returns the bilinear point and the corresponding weight
"""
import math
def bilinear_interpolation(coordinate, original_size, scale):
    """
    @param coordinate: the coordinate in the output space, needs to compute. --- (x, y)
    @param original_size: the original size of height and width --- (h, w)
    @param scale: the scale factor between the output and input
    @return: dic{coordinate(u, v): weight}. returns the corresponding coordinate in the input space and there corresponding weights
    method:
        first compute the coordinate of point in the input space by: r = (x + 1) / scale + 1 / (2 * scale) - 1/2
                                                                     c = (y + 1) / scale + 1 / (2 * scale) - 1/2
        then let: u = floor_int(r), v = floor_int(c). delta_r = r - u, delta_c = c - v
        having that: (r, c) with weights (1-delta_r)*(1-delta_c); (r+1, c) with weights delta_r * (1-delta_c); (r, c+1) with weights (1-delta_r) * delta_c; (r+1, c+1) with weights delta_r * delta_c
        note that: if u is out of boundary, delta_r = 1; if u + 1 is out of boundary, then delta_r = 0. same to c
    """
    x = coordinate[0]
    y = coordinate[1]
    h = original_size[0]
    w = original_size[1]
    r = x / scale + 1.0 / (2.0 * scale) - 0.5
    c = y / scale + 1.0 / (2.0 * scale) - 0.5
    u = math.floor(r)
    v = math.floor(c)
    delta_r = r - u
    delta_c = c - v
    if u < 0:
        delta_r = 1
    if u + 1 >= h:
        delta_r = 0
    if v < 0:
        delta_c = 1
    if v + 1 >= w:
        delta_c = 0
    result = {}
    if (1 - delta_r) * (1 - delta_c) != 0:
        result[(u, v)] = (1-delta_r) * (1 - delta_c)
    if delta_r * (1 - delta_c) != 0:
        result[(u+1, v)] = delta_r * (1 - delta_c)
    if (1 - delta_r) * delta_c != 0:
        result[(u, v+1)] = (1 - delta_r) * delta_c
    if delta_r * delta_c != 0:
        result[(u+1, v+1)] = delta_r * delta_c

    return result


