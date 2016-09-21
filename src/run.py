import caffe
import os.path as osp
import matplotlib.pyplot as plt
from tools import SimpleTransformer as ST
import numpy as np
import scipy.misc
from PIL import Image
import time

print ('init nerwork...')
start_time = time.time()
modeldir = '../model'
model = osp.join(modeldir, 'deploy.prototxt')
weigths = osp.join(modeldir, 'batch5_iter_60000.caffemodel')
caffe.set_mode_cpu()
#caffe.set_device(0)
net = caffe.Net(model, weigths, caffe.TEST)
end_time = time.time()
print "net init done. Time consumes ", (end_time - start_time), "seconds"


# prepare the data
im = np.asarray(Image.open('/home/qinhong/workdir/project/cmu_project/data/val_data/test1.jpg'))
w = im.shape[0]
h = im.shape[1]
im = scipy.misc.imresize(im, (224, 224))
im = ST().preprocess(im)

print ('begin forward')
start_time = time.time()
net.blobs['data'].data[...] = im
net.forward()
predict = net.blobs['fc3'].data
l2 = np.linalg.norm(predict, axis=1)
normed_predict = predict / l2[:, None]
normed_predict = np.reshape(normed_predict, (224,224,3))
end_time = time.time()
print "forward consumes %s", (end_time - start_time), "seconds"

plt.figure(1)
predict = scipy.misc.imresize(np.uint8((normed_predict/2 + 0.5) * 255), (w, h))
plt.imshow(predict)
plt.show()

#solver.test_nets.copy_from(osp.join(modeldir, 'tune_iter_250.caffemodel'))
# the blobs shape
#print "For blobs shape---------------------------------------"
#for layer_name, blob in solver.net.blobs.iteritems():
#    print layer_name + '\t' + str(blob.data.shape)
# the params shape
#print "\n\n\nFor params shape--------------------------------------"
#for layer_name, param in solver.net.params.iteritems():
#    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

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
#for it in range(50):
#    solver.net.forward()
#     """
#   plt.figure(1)
#   data = solver.net.blobs['data'].data[0,...]
#    plt.imshow(ST().deprocess(data))
#    #plt.imshow((solver.net.blobs['normal'].data[0,...]).transpose(1,2,0))
#    plt.figure(2)
#    normal = solver.net.blobs['normal'].data[0,...].transpose(1,2,0)
#    normal = np.uint8((normal / 2 + 0.5) * 255)
#    plt.imshow(normal)
#    plt.show()
#    """
#print 'debugging'