% This code is for ECE271A HW2. Create date: Oct/22/2020.
close all;
%% Setup
Train_sample = load('TrainingSamplesDCT_8_new.mat');
Trainbg = Train_sample.TrainsampleDCT_BG;
Trainfg = Train_sample.TrainsampleDCT_FG;
%% Problem (a) prior probability
Py_cheetah = size(Trainfg,1)/( size(Trainfg,1)+size(Trainbg,1) );
Py_grass = 1 - Py_cheetah;
%% Problem (b) 64 plots
bg_mean = sum(Trainbg)/size(Trainbg,1);
fg_mean = sum(Trainfg)/size(Trainfg,1);

bg_std = std(Trainbg);
fg_std = std(Trainfg);
overlap_portion = zeros(64,1);

f1 = figure(1);
for i = 1 : 64
    x_s = min(bg_mean(1,i) - bg_std(1,i)*4 , fg_mean(1,i) - fg_std(1,i)*4);
    x_e = max(bg_mean(1,i) + bg_std(1,i)*4 , fg_mean(1,i) + fg_std(1,i)*4);
    x_axis = linspace(x_s,x_e);
    
    ybg = myGaussain(x_axis, bg_mean(1,i),bg_std(1,i));
    yfg = myGaussain(x_axis, fg_mean(1,i),fg_std(1,i));
    subplot(8,8,i);
    plot(x_axis, ybg, '--', x_axis, yfg, ':');

    hold on;
    minY = min(ybg,yfg);
    area(x_axis,minY);
    overlap = cumtrapz(x_axis,minY);
    overlap_portion(i) = overlap(end);
    
    title(i);
end
legend('background','foreground','Location','northeast');
legend('boxoff');
saveas(f1,'g64.png');
%% Problem (b) : best and worst 8 features
[~,bestindex] = mink(overlap_portion,8);
[~,worstindex] = maxk(overlap_portion,8);
f2 = figure(2);
subploti = 1;
for i = bestindex'
    
    x_s = min(bg_mean(1,i) - bg_std(1,i)*4 , fg_mean(1,i) - fg_std(1,i)*4);
    x_e = max(bg_mean(1,i) + bg_std(1,i)*4 , fg_mean(1,i) + fg_std(1,i)*4);
    x_axis = linspace(x_s,x_e);
    
    ybg = myGaussain(x_axis, bg_mean(1,i),bg_std(1,i));
    yfg = myGaussain(x_axis, fg_mean(1,i),fg_std(1,i));
    subplot(4,2,subploti);
    plot(x_axis, ybg, '--', x_axis, yfg, ':');
    subploti = subploti +1;
    hold on;
    minY = min(ybg,yfg);
    area(x_axis,minY);
    title(i);
end
legend('background','foreground');
legend('boxoff');
saveas(f2,'b8.png');
f3 = figure(3);
subploti = 1;
for i = worstindex'
    x_s = min(bg_mean(1,i) - bg_std(1,i)*4 , fg_mean(1,i) - fg_std(1,i)*4);
    x_e = max(bg_mean(1,i) + bg_std(1,i)*4 , fg_mean(1,i) + fg_std(1,i)*4);
    x_axis = linspace(x_s,x_e);
    
    ybg = myGaussain(x_axis, bg_mean(1,i),bg_std(1,i));
    yfg = myGaussain(x_axis, fg_mean(1,i),fg_std(1,i));
    subplot(4,2,subploti);
    plot(x_axis, ybg, '--', x_axis, yfg, ':');
    hold on;
    subploti = subploti +1;
    minY = min(ybg,yfg);
    area(x_axis,minY);
    title(i);
end
legend('background','foreground');
legend('boxoff');
saveas(f3,'w8.png');
%% Problem (c)

imaC = im2single(imread('cheetah.bmp'));
imaC_pad = padarray(imaC,[1,2],'replicate','post');
Path = load('Zig-Zag Pattern.txt');
Path = Path+1;
current_blocks = zeros(1,64);
CheetahMask = zeros(size(imaC_pad,1),size(imaC_pad,2));
%% Problem (c) :  using all features

cov_all_fg = cov(Trainfg);
cov_all_bg = cov(Trainbg);
cov_best_fg = cov(Trainfg(:,bestindex));
cov_best_bg = cov(Trainbg(:,bestindex));

for i = 1 : size(imaC_pad,1)-7
    for j = 1 : size(imaC_pad,2)-7
        current_dc2_blocks = dct2(imaC_pad(i:i+7,j:j+7));
        current_blocks(Path) = current_dc2_blocks;
        M_C = mymvn(current_blocks, fg_mean, cov_all_fg);
        M_G = mymvn(current_blocks, bg_mean, cov_all_bg);
        if (M_C * Py_cheetah) > (M_G * Py_grass)
            CheetahMask(i,j) = 1;
        else
            CheetahMask(i,j) = 0;
        end
    end
end
f4 = figure(4);
imagesc(CheetahMask);
colormap(gray(255));
saveas(f4,'cheetah64.png');
imay = im2single(imread("cheetah_mask.bmp"));
imay_pad = padarray(imay,[1,2],'replicate','post');
True_cheetah = sum(imay_pad,'all');
True_grass = sum(abs(imay_pad-1), 'all');
[detection64, false_alarm64] = error(CheetahMask, imay_pad);
error_rate64 = false_alarm64/True_grass * Py_grass + (1-detection64/True_cheetah) * Py_cheetah;
%% Problem (c) :  using 8 features
for i = 1 : size(imaC_pad,1)-7
    for j = 1 : size(imaC_pad,2)-7
        current_dc2_blocks = dct2(imaC_pad(i:i+7,j:j+7));
        current_blocks(Path) = current_dc2_blocks;
        M_C = mymvn(current_blocks(bestindex), fg_mean(bestindex), cov_best_fg);
        M_G = mymvn(current_blocks(bestindex), bg_mean(bestindex), cov_best_bg);
        if (M_C * Py_cheetah) > (M_G * Py_grass)
            CheetahMask(i,j) = 1;
        else
            CheetahMask(i,j) = 0;
        end
    end
end
f5 = figure(5);
imagesc(CheetahMask);
colormap(gray(255));
saveas(f5,'cheetah8.png');
[detection8, false_alarm8] = error(CheetahMask, imay_pad);
error_rate8 = false_alarm8/True_grass * Py_grass + (1-detection8/True_cheetah) * Py_cheetah;
%% Function

function [error_c,error_g] = error(pred, true)
    error_c = sum(pred.*true,'all');
    error_g = 0;
    for i = 1 : size(pred,1)
        for j = 1 : size(pred,2)
            if true(i,j) == 0
                if pred(i,j) == 1
                    error_g = error_g + 1;
                end
            end
        end
    end
end

function y = myGaussain(x,mean,std)
    constant = std*sqrt(2*pi);
    y = exp(-0.5*((x-mean)/std).^2) / constant;
end

function y = mymvn(x,mean,cov)
    constant = sqrt(det(cov)*(2*pi)^(size(cov,2)));
    exp_component = -0.5 * (x-mean) /cov * (x-mean)';
    y = exp(exp_component)/constant;
end