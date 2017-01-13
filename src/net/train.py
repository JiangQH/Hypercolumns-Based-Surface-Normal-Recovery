import caffe
import os.path as osp
import matplotlib.pyplot as plt
from tools import SimpleTransformer as ST
import numpy as np

modeldir = '../model'
caffe.set_mode_cpu()
solver = caffe.SGDSolver(osp.join(modeldir, 'solver.prototxt'))
solver.net.copy_from(osp.join(modeldir, 'vgg_16_full_conv.caffemodel'))
solver.solve()
