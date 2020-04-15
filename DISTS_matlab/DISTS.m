% This is a matlab implementation of calculating the
% Deep Image Structure and Texture Similarity (DISTS) between two images.
% https://github.com/dingkeyan93/DISTS
% Requirements: Matlab>=2019b

% Input: 
% (1) img1: the first image being compared (range:0~255)
% (2) img2: the second image being compared (range:0~255)
% (3) params: the pretrained vgg16 parameters
% (4) weights: the trained perceptual weights
% (5) resize_img: if resize the input image to 256 (short side)
% (6) use_gpu: if use GPU to accelerate
% Output:
% (1) perceptual quality score between two images (smaller is better)

% Example:
% ref = imread('../images/r0.png');
% dist = imread('../images/r1.png');
% net_params = load('../weights/net_param.mat');
% weights = load('../weights/alpha_beta.mat');
% resize_img = 0; 
% use_gpu = 0;
% score = DISTS(ref, dist,net_params,weights, resize_img, use_gpu) % 0.3347

% Note: 
% The results of some images are a little different from the python version. 
% Be suject to the pytorch version.

function score = DISTS(ref, dist, params, weights, resize, gpu)
    ref_features = extract_features(ref,params,resize,gpu);
    dist_features = extract_features(dist,params,resize,gpu);
    dist1 = 0;
    dist2 = 0;
    c1 = 1e-6;
    c2 = 1e-6;
    chns = [3,64,128,256,512,512];
    alpha = split_weights(weights.alpha,chns);
    beta = split_weights(weights.beta,chns);
    % weights_sum = sum(weights.alpha+weights.beta);
    for i = 1:6
        ref_mean = mean(ref_features{i},[1,2]);
        dist_mean = mean(dist_features{i},[1,2]);
        ref_var = mean((ref_features{i}-ref_mean).^2,[1,2]);
        dist_var = mean((dist_features{i}-dist_mean).^2,[1,2]);
        ref_dist_cov = mean(ref_features{i}.*dist_features{i},[1,2])-ref_mean.*dist_mean;
        S1 = (2*ref_mean.*dist_mean+c1)./(ref_mean.^2+dist_mean.^2+c1);
        S2 = (2*ref_dist_cov+c2)./(ref_var+dist_var+c2);
        dist1 = dist1+sum(alpha{i}.*S1.squeeze());
        dist2 = dist2+sum(beta{i}.*S2.squeeze());
    end
    score = extractdata(1-(dist1+dist2));
    score = gather(score);
end

function features = extract_features(I, params, resize, gpu)
    if resize && min(size(I,1),size(I,2))>256
        I = imresize(I,256/min(size(I,1),size(I,2)));
    end
    if gpu
        I = gpuArray(I);
    end
    I = dlarray(double(I)/255,'SSC');
    
    features = cell(6,1);
    % stage 0
    features{1} = I;
    dlX = (I - params.vgg_mean)./params.vgg_std;
    
    % stage 1
    weights = dlarray(params.conv1_1_weight);
    bias = dlarray(params.conv1_1_bias');
    dlY = relu(dlconv(dlX,weights,bias,'Stride',1,'Padding','same'));
    
    weights = dlarray(params.conv1_2_weight);
    bias = dlarray(params.conv1_2_bias');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    features{2} = dlY;
    
    % stage 2
    weights = dlarray(params.L2pool_1);
    dlY = dlconv(dlY.^2,weights,0,'Stride',2,'Padding',[1, 1; 0, 0]);
    dlY = sqrt(dlY);
    % dlY = avgpool(dlY,2,'Stride',2);
    
    weights = dlarray(params.conv2_1_weight);
    bias = dlarray(params.conv2_1_bias');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    weights = dlarray(params.conv2_2_weight);
    bias = dlarray(params.conv2_2_bias');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    features{3} = dlY;
    
    % stage 3
    weights = dlarray(params.L2pool_2);
    dlY = dlconv(dlY.^2,weights,0,'Stride',2,'Padding',[1, 1; 0, 0]);
    dlY = sqrt(dlY);
    % dlY = avgpool(dlY,2,'Stride',2);
    
    weights = dlarray(params.conv3_1_weight);
    bias = dlarray(params.conv3_1_bias');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    weights = dlarray(params.conv3_2_weight);
    bias = dlarray(params.conv3_2_bias');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    weights = dlarray(params.conv3_3_weight);
    bias = dlarray(params.conv3_3_bias');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    features{4} = dlY;
    
    % stage 4
    weights = dlarray(params.L2pool_3);
    dlY = dlconv(dlY.^2,weights,0,'Stride',2,'Padding',[1, 1; 0, 0]);
    dlY = sqrt(dlY);
    % dlY = avgpool(dlY,2,'Stride',2);
    
    weights = dlarray(params.conv4_1_weight);
    bias = dlarray(params.conv4_1_bias');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    weights = dlarray(params.conv4_2_weight);
    bias = dlarray(params.conv4_2_bias');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    weights = dlarray(params.conv4_3_weight);
    bias = dlarray(params.conv4_3_bias');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    features{5} = dlY;
    
    % stage 5
    weights = dlarray(params.L2pool_4);
    dlY = dlconv(dlY.^2,weights,0,'Stride',2,'Padding',[1, 1; 0, 0]);
    dlY = sqrt(dlY);
    % dlY = avgpool(dlY,2,'Stride',2);
    
    weights = dlarray(params.conv5_1_weight);
    bias = dlarray(params.conv5_1_bias');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    weights = dlarray(params.conv5_2_weight);
    bias = dlarray(params.conv5_2_bias');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    weights = dlarray(params.conv5_3_weight);
    bias = dlarray(params.conv5_3_bias');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    features{6} = dlY;
end

function w_ = split_weights(w,chns)
    w_ = cell(length(chns),1);
    for i=1:length(chns)
        w_{i}=w(1:chns(i))';
        w(1:chns(i))=[];
    end
end

