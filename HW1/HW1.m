% This code is for ECE271A HW1. Create date: Oct/6/2020.
close all;
%% Setup
%Load the files;
Train_sample = load('TrainingSamplesDCT_8.mat');
Trainbg = Train_sample.TrainsampleDCT_BG;
Trainfg = Train_sample.TrainsampleDCT_FG;
%----------------------------------------------------------------------%
%% Problem (a)
% Calculate the probability of Py(Cheetah) and Py(grass)
Py_cheetah = size(Trainfg,1)/( size(Trainfg,1)+size(Trainbg,1) );
Py_grass = 1 - Py_cheetah;
%----------------------------------------------------------------------%
%% Problem (b)
% Setting the largest coefficient as a dummy variable.
Trainbg(:,1) = -1;
Trainfg(:,1) = -1;
% Finds out second largest coefficient's index
[~,bg_index] = max(Trainbg,[],2);
[~,fg_index] = max(Trainfg,[],2);
bincount_bg_index = histcounts(bg_index);
Pg = bincount_bg_index/size(Trainbg,1);
f1 = figure('name','Problem (b)_2');
bar(Pg,0.4);
title('\fontsize{14}Px|y(x|Grass)');
bincount_fg_index = histcounts(fg_index);
Pc = bincount_fg_index/size(Trainfg,1);
f2 = figure('name','Problem (b)_1');
bar(Pc,0.4);
title('\fontsize{14}Px|y(x|Cheetah)');
Pc = padarray(Pc,[0,32],0,'post');
Pg = padarray(Pg,[0,47],0,'post');
%----------------------------------------------------------------------%
%% Problem (c)
imaC = im2single(imread('cheetah.bmp'));
imaC_pad = padarray(imaC,[1,2],'replicate','post');
Path = load('Zig-Zag Pattern.txt');
Path = Path+1;
current_blocks = zeros(8,8);
CheetahMask = zeros(size(imaC_pad,1),size(imaC_pad,2));
for i = 1 : size(imaC_pad,1)-7
    for j = 1 : size(imaC_pad,2)-7
        current_dc2_blocks = dct2(imaC_pad(i:i+7,j:j+7));
        current_blocks(Path) = current_dc2_blocks;
        [~,index] = max(abs(current_blocks(2:64)));
        if Pc(index) * Py_cheetah > Pg(index) * Py_grass
            CheetahMask(i,j) = 1;
        else
            CheetahMask(i,j) = 0;
        end
    end
end
f3 = figure(3);
imagesc(CheetahMask);
colormap(gray(255));
imaC_test = imaC_pad + CheetahMask;
imagesc(imaC_test);
%saveas(f1,'1.png')
%saveas(f2,'2.png')
saveas(f3,'3.png')
%----------------------------------------------------------------------%
%% Problem(d)
%----------------------------------------------------------------------%
imay = im2single(imread("cheetah_mask.bmp"));
imay_pad = padarray(imay,[1,2],'replicate','post');
mask_difference = sum(abs(CheetahMask - imay_pad),'all');
error_Mask = mask_difference/(256*272);
% coded by Jaw-Yuan, Chang
