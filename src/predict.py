import caffe
import os.path as osp
from tools import SimpleTransformer as ST
import numpy as np
import scipy.misc
from PIL import Image
import time
import sys

def prediction(model_file, model_weights, img_path):
	print 'init network...'
	start_time = time.time()
	if not osp.exists(model_file):
		raise ValueError("can not find deploy.prototxt")
	if not osp.exists(model_weights):
		raise ValueError("can not find .caffemodel weights file")
	caffe.set_mode_gpu()
	caffe.set_device(0)
	net = caffe.Net(model_file, model_weights, caffe.TEST)
	print 'network init done, time consumes {} seconds'.format(time.time() - start_time)

	im = np.asarray(Image.open(img_path))
	w = im.shape[0]
	h = im.shape[1]
	im = scipy.misc.imresize(im, (224, 224))
	im = ST().preprocess(im)
	normal = np.zeros_like(im)
	net.blobs['data'].data[...] = im
	net.blobs['normal'].data[...] = normal
	print 'net begin forward...'
	start_time = time.time()
	net.forward()
	predict = net.blobs['fc3'].data

	"""
	Note that below code is needed only when you set the skip ratio in the deploy file's
	hypercolumns layer parameter. the skip ratio is used to do acceleration when doing the prediction
	skip some points. the skip ratio here should be same with that one in the file

	skip_ratio = 4
	N = predict.shape[0]
	X = np.arange(0, skip_ratio*N, skip_ratio)
	X_new = np.arange(skip_ratio*N)
	y0 = np.reshape(np.interp(X_new, X, predict[:,0]),(-1,1))
	y1 = np.reshape(np.interp(X_new, X, predict[:,1]),(-1,1))
	y2 = np.reshape(np.interp(X_new, X, predict[:,2]),(-1,1))
	predict = np.concatenate((y0,y1,y2), axis=1)
	"""

	l2 = np.linalg.norm(predict, axis=1)
	normed_predict = predict / l2[:, None]
	normed_predict = np.reshape(normed_predict, (224,224,3))
	end_time = time.time()
	print "forward consumes ", (end_time - start_time), "seconds"
	predict = scipy.misc.imresize(np.uint8((normed_predict/2 + 0.5) * 255), (w, h))
	file_name, extension = osp.splitext(img_path)
	scipy.misc.imsave(file_name+'_prediction'+extension, predict)
	print 'saved the prediction as {}'.format(file_name+'_prediction'+extension)


if __name__ == '__main__':
	if len(sys.argv) != 4:
		raise ValueError("There should be exactly 3 params passed. ---/path/to/deploy.prototxt /path/to/weights.caffemodel /path/to/img")
	model_file = sys.argv[1]
	model_weights = sys.argv[2]
	img_path = sys.argv[3]
	prediction(model_file, model_weights, img_path)