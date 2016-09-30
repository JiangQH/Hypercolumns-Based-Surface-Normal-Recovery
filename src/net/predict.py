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
modeldir = '../model/origin'
model = osp.join(modeldir, 'deploy.prototxt')
#weigths = '../model/finetune_full_conv.caffemodel'
weigths = '../model/origin/batch7_original.caffemodel'
caffe.set_mode_gpu()
caffe.set_device(2)
net = caffe.Net(model, weigths, caffe.TEST)
end_time = time.time()
print "net init done. Time consumes ", (end_time - start_time), "seconds"


# prepare the data
#im = np.asarray(Image.open('/home/qinhong/project/normal_recovery/data/outdoor/stereo/val/391.png'))
#w = im.shape[0]
#h = im.shape[1]
#im = scipy.misc.imresize(im, (224, 224))
#im = ST().preprocess(im)
#normal = np.zeros_like(im)


im = np.asarray(Image.open('/home/qinhong/project/normal_recovery/src/test1.jpg'))
w = im.shape[0]
h = im.shape[1]
im = scipy.misc.imresize(im, (224, 224))
im = ST().preprocess(im)
normal = np.zeros_like(im)
net.blobs['data'].data[...] = im
net.blobs['normal'].data[...] = normal
print ('begin forward')
start_time = time.time()
net.forward()

predict = net.blobs['fc3'].data
N = predict.shape[0]
X = np.arange(0, 4*N, 4)
X_new = np.arange(4*N)
y0 = np.reshape(np.interp(X_new, X, predict[:,0]),(-1,1))
y1 = np.reshape(np.interp(X_new, X, predict[:,1]),(-1,1))
y2 = np.reshape(np.interp(X_new, X, predict[:,2]),(-1,1))
predict = np.concatenate((y0,y1,y2), axis=1)
#predict = np.append([y0],[y1],[y2], axis=1)
l2 = np.linalg.norm(predict, axis=1)
normed_predict = predict / l2[:, None]
normed_predict = np.reshape(normed_predict, (224,224,3))
end_time = time.time()
print "forward consumes ", (end_time - start_time), "seconds"


predict = scipy.misc.imresize(np.uint8((normed_predict/2 + 0.5) * 255), (w, h))
scipy.misc.imsave('8.png', predict)
#plt.figure(1)
#predict = scipy.misc.imresize(np.uint8((normed_predict/2 + 0.5) * 255), (w, h))
#plt.imshow(predict)
#plt.show()

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
