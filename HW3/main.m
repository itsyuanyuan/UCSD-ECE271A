% This code is for ECE271A HW3. Create date: Nov/19/2020.
%% Loading data
close all; clear all;
load('TrainingSamplesDCT_subsets_8.mat');
P1 = load('Prior_1.mat');
P2 = load('Prior_2.mat');
load('Alpha.mat');
Path = load('Zig-Zag Pattern.txt');
Path = Path+1;
imaC = im2single(imread('cheetah.bmp'));
imaC_pad = padarray(imaC,[1,2],'replicate','post');
current_blocks = zeros(1,64);
DCT_blocks = zeros(size(imaC_pad,1),size(imaC_pad,2),64);
CheetahMask = zeros(size(imaC_pad,1),size(imaC_pad,2));
%%
imay = im2single(imread("cheetah_mask.bmp"));
imay_pad = padarray(imay,[1,2],'replicate','post');
%% error matrix
error_bayes = zeros(1,9);
error_MAP = zeros(1,9);
%% DCT Blocks
for k = 1 : size(imaC_pad,1)-7
    for j = 1 : size(imaC_pad,2)-7
        current_dc2_blocks = dct2(imaC_pad(k:k+7,j:j+7));
        current_blocks(Path) = current_dc2_blocks;
        DCT_blocks(k,j,:) = current_blocks(:);
    end
end
%% Data1, Stregy 1
[m_bg, cov_bg] = getmcov(D4_BG);
[m_fg, cov_fg] = getmcov(D4_FG);

Py_d1c = size(D1_FG,1)/(size(D1_FG,1)+size(D1_BG,1));
Py_d1g = 1-Py_d1c;
%%
index = 1;
for i = alpha
    cov0 = i * diag(P2.W0);
    [b_m_b, b_c_b] = getbayesmcov(size(D4_BG,1),cov0,cov_bg,m_bg,P2.mu0_BG');
    [b_m_f, b_c_f] = getbayesmcov(size(D4_FG,1),cov0,cov_fg,m_fg,P2.mu0_FG');
    
    % Predictive
    m_pred_bg = b_m_b;
    cov_pred_bg = b_c_b + cov_bg;
    m_pred_fg = b_m_f;
    cov_pred_fg = b_c_f + cov_fg;

    for k = 1 : size(imaC_pad,1)-7
        for j = 1 : size(imaC_pad,2)-7
            dct = DCT_blocks(k,j,:);
            dct = squeeze(dct);

            M_C = mymvn(dct,m_pred_fg ,cov_pred_fg);
            M_G = mymvn(dct,m_pred_bg ,cov_pred_bg);
            if (M_C * Py_d1c) > (M_G * Py_d1g)
                CheetahMask(k,j) = 1;
            else
                CheetahMask(k,j) = 0;
            end
        end
    end
    error_bayes(index) = error(CheetahMask,imay_pad);
    % MAP

    for k = 1 : size(imaC_pad,1)-7
        for j = 1 : size(imaC_pad,2)-7
            dct = DCT_blocks(k,j,:);
            dct = squeeze(dct);
            M_C = mymvn(dct,b_m_f ,cov_fg);
            M_G = mymvn(dct,b_m_b ,cov_bg);
            if (M_C * Py_d1c) > (M_G * Py_d1g)
                CheetahMask(k,j) = 1;
            else
                CheetahMask(k,j) = 0;
            end
        end
    end
    error_MAP(index) = error(CheetahMask,imay_pad);
    index = index +1;
    disp(error_bayes);
    disp(error_MAP);
end
%% ML

for k = 1 : size(imaC_pad,1)-7
    for j = 1 : size(imaC_pad,2)-7
        dct = DCT_blocks(k,j,:);
        dct = squeeze(dct);
        M_C = mymvn(dct,m_fg ,cov_fg);
        M_G = mymvn(dct,m_bg ,cov_bg);
        if (M_C * Py_d1c) > (M_G * Py_d1g)
            CheetahMask(k,j) = 1;
        else
            CheetahMask(k,j) = 0;
        end
    end
end
eML = error(CheetahMask, imay_pad);
%%
f1 = figure(1);
error_ML = eML * ones(1,9);
semilogx(alpha, error_bayes*100, 'r-o', alpha, error_MAP*100, 'g-o', alpha, error_ML*100, 'b-o');
title(' Classification error');
xlabel('alpha'); ylabel('error(%)');
legend('Predictive', 'MAP', 'ML');
saveas(f1,'D4_S2.png')
%% Function
function [b_m, b_c] = getbayesmcov(n,cov0,cov,un,u0)
    b_m = cov0 / (cov0 + cov/n) * un + (cov/(cov0 + cov/n))/n *u0 ;
    b_c = cov0/(cov0 + cov/n) * (cov/n);
end

function [mean ,cov] = getmcov(x)
    mean = (x'*ones(size(x,1),1))/size(x,1);
    cov = x'*x/size(x,1) - mean*mean';
end

function error = error(pred, true)
    error = sum(abs(pred-true),'all')/(256*272);
end

function y = mymvn(x,mean,cov)
    constant = sqrt(det(cov)*(2*pi)^(size(cov,2)));
    exp_component = -0.5 * (x-mean)' /(cov) * (x-mean);
    y = exp(exp_component)/constant;
end
