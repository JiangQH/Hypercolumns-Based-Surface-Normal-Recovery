"""
this file is aimed to read the binary normal. and save it as figure
"""
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
normal_file = '../../eval/indoor/1449.txt'
with open(normal_file, 'rb') as fid:
    data_array = np.fromfile(fid, np.float32)
normal = np.zeros((480,640,3))
count = 0
for kk in range(3):
    for ww in range(640):
        for hh in range(480):
            normal[hh, ww, kk] = data_array[count]
            count += 1
normal_pic = np.uint8((normal/2 + 0.5) * 255)
scipy.misc.imsave('gt1449.png',normal_pic)