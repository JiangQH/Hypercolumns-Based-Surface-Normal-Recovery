function prepared = prepare_image(im, au)
    % im color image as uint8 H*W*3
    % au whether do the augumentation. sub the mean data
    % prepare the image to caffe's input of W * H * 3 with BGR channels
    IMAGE_DIM = 224;
    %% whether do the data augumentation. in BGR mode
    b_mean = 104;
    g_mean = 117;
    r_mean = 123;
    mean_data = ones(IMAGE_DIM, IMAGE_DIM, 3);
    mean_data(:,:,1) = mean_data(:,:,1) * b_mean;
    mean_data(:,:,2) = mean_data(:,:,2) * g_mean;
    mean_data(:,:,3) = mean_data(:,:,3) * r_mean;
    %% do the prepare work
    im_data = im(:,:,[3, 2, 1]); % permute channels from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]); % flip width and height
    im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear'); % resize the image
    im_data = single(im_data); % convert to single
    if au
        im_data = im_data - mean_data; % subtract the mean_data
    end
    
    prepared = im_data;
     
end