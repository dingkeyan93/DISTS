clear
ref = imread('../images/r0.png');
dist = imread('../images/r1.png');
net_params = load('../weights/net_param.mat');
weights = load('../weights/alpha_beta.mat');
resize_img = 0; 
use_gpu = 0;
score = DISTS(ref, dist,net_params,weights, resize_img, use_gpu)