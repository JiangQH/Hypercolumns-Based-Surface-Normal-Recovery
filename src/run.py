import caffe
import os.path as osp
import matplotlib.pyplot as plt
from tools import SimpleTransformer as ST
import numpy as np
modeldir = '../model'
caffe.set_mode_cpu()
solver = caffe.SGDSolver(osp.join(modeldir, 'solver.prototxt'))
solver.net.copy_from(osp.join(modeldir, 'VGG_16_full_conv.caffemodel'))
#solver.test_nets.copy_from(osp.join(modeldir, 'tune_iter_250.caffemodel'))
# the blobs shape
print "For blobs shape---------------------------------------"
for layer_name, blob in solver.net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)
# the params shape
print "\n\n\nFor params shape--------------------------------------"
for layer_name, param in solver.net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
"""
# visulize something
solver.net.forward()
# the image
image = solver.net.blobs['data'].data
#image += [104,117,123]
vis_inter(image.transpose(0, 2, 3, 1), 0)
# the normal
normal = solver.net.blobs['normal'].data
vis_inter(normal.transpose(0, 2, 3, 1), 1)
plt.show()
"""
for it in range(50):
    solver.step(1)
    """
    plt.figure(1)
    data = solver.net.blobs['data'].data[0,...]
    plt.imshow(ST().deprocess(data))
    #plt.imshow((solver.net.blobs['normal'].data[0,...]).transpose(1,2,0))
    plt.figure(2)
    normal = solver.net.blobs['normal'].data[0,...].transpose(1,2,0)
    normal = np.uint8((normal / 2 + 0.5) * 255)
    plt.imshow(normal)
    plt.show()
    """
print 'debugging'