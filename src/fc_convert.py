import caffe
# load the vgg net
vgg_16 = caffe.Net('../model/vgg_16/VGG_16_deploy.prototxt',
                   '../model/vgg_16/VGG_ILSVRC_16_layers.caffemodel',
                   caffe.TEST)
params = ['fc6', 'fc7', 'fc8']
fc_params = {pr: (vgg_16.params[pr][0].data, vgg_16.params[pr][1].data) for pr in params}
for fc in fc_params:
    print '{} weights are {} dimensional and bias are {} dimentional'.format(fc, fc_params[fc][0].shape,
                                                                         fc_params[fc][1].shape)
#print [(k, v.data.shape) for k, v in vgg_16.blobs.items()]
print [(k, v.data.shape) for k, v in vgg_16.blobs.items()]
# the fully-conv net
vgg_16_full_conv = caffe.Net('../model/vgg_16_full_conv.prototxt',
                             '../model/vgg_16/VGG_ILSVRC_16_layers.caffemodel',
                             caffe.TEST)
fc_conv_params = ['fc6-conv', 'fc7-conv', 'fc8-conv']
conv_params = {pr: (vgg_16_full_conv.params[pr][0].data,
                    vgg_16_full_conv.params[pr][1].data) for pr in fc_conv_params}
for fc in conv_params:
    print '{} weights are {} dimensional and bias are {} dimentional'.format(fc, conv_params[fc][0].shape,
                                                                             conv_params[fc][1].shape)
for pr, pr_conv in zip(params, fc_conv_params):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat
    conv_params[pr_conv][1][...] = fc_params[pr][1]
vgg_16_full_conv.save('../model/VGG_16_full_conv.caffemodel')
