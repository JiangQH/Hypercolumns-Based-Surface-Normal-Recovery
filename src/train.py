import caffe
import os.path as osp
import matplotlib.pyplot as plt
from tools import SimpleTransformer as ST
import numpy as np

modeldir = '../model'
caffe.set_mode_cpu()
solver = caffe.SGDSolver(osp.join(modeldir, 'solver.prototxt'))
solver.net.copy_from(osp.join(modeldir, 'batch5_iter_60000.caffemodel'))
for i in range(100):
    solver.step(1)
    plt.figure(1)
    data = solver.net.blobs['data'].data[0, ...]
    plt.imshow(ST().deprocess(data))
   # plt.imshow((solver.net.blobs['normal'].data[0,...]).transpose(1,2,0))
    plt.figure(2)
    normal = solver.net.blobs['normal'].data[0, ...].transpose(1, 2, 0)
    normal = np.uint8((normal / 2 + 0.5) * 255)
    plt.imshow(normal)
    plt.show()
