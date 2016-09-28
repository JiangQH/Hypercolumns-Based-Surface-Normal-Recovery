%% set the network 
model = '../model/deploy.prototxt';
weights = '../model/deploy.caffemodel';
caffe.set_mode_gpu();
caffe.set_device(0);
net = caffe.Net(model, weights, 'test');

%% set the input data
width = 224;
height = 224;
im_data = caffe.io.load_image('/home/qinhong/workdir/project/cmu_project/data/val_data/108.png');
im_data = imresize(im_data, [width, height]);
% im = imread('/home/qinhong/workdir/project/cmu_project/data/val_data/108.png');
% im_data = prepare_image(im, true);
%% do data augumentation?


%% output
prediction = net.forward({im_data});
prediction = prediction{1};
normals = zeros(224, 224, 3);
normals(:,:,1) = reshape(prediction(1,:), 224, 224);
normals(:,:,2) = reshape(prediction(2,:), 224, 224);
normals(:,:,3) = reshape(prediction(3,:), 224, 224);
