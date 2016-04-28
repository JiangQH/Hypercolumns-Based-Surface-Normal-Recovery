#!/usr/bin/env python
# coding=utf-8
import os
import os.path as osp
def select_image(img_dir):
    """
    this function will return the selected rgb and depth name
    """
    rgb_images = []
    rgb_images += [rgb for rgb in os.listdir(img_dir) if rgb.endswith('.ppm')]
    # note the image in nyu nearly have 30 images per head name, so here sample distance adpot 10
    # we will get 40k images somehow
    rgb_images.sort()
    result = []
    for rgb in rgb_images:
        base_name = osp.splitext(rgb)
        depth = osp.join(base_name, '.pgm')
        result.append([rgb, depth])
    return result

def random_select(data_dir):
    """
    this function is the main function used to 
    randomly select image and depth from the nyu dataset
    """
    abs = osp.abspath(data_dir)
    rgb_des_dir = osp.join(abs, 'selected/rgb')
    depth_des_dir = osp.join(abs, 'selected/depth')
    if not osp.isdir(rgb_des_dir):
        os.makedirs(rgb_des_dir)
    if not osp.isdir(depth_des_dir):
        os.makedirs(depth_des_dir)

    selected_truple = []
    type_dirs = os.listdir(abs)
    for type in type_dirs:
        scene_dirs = os.listdir(type)
        for scene in scene_dirs:
            selected_truple.append(select_image(osp.abspath(scene)))
    

    for [rgb, depth] in selected_truple:
       rgb_base_name = osp.basename(rgb)
       depth_base_name = osp.basename(depth)
       des_rgb = osp.join(rgb_des_dir, rgb_base_name)
       des_depth = osp.join(depth_des_dir, depth_base_name)
       os.rename(rgb, des_rgb)
       os.rename(depth, des_depth)

