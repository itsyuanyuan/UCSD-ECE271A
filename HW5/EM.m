close all; clear all;
%% Loading Data and some preparation
load('TrainingSamplesDCT_8_new.mat');
Path = load('Zig-Zag Pattern.txt');
Path = Path+1;
imaC = im2single(imread('cheetah.bmp'));
imaC_pad = padarray(imaC,[1,2],'replicate','post');
imay = im2single(imread("cheetah_mask.bmp"));
imay_pad = padarray(imay,[1,2],'replicate','post');
current_blocks = zeros(1,64);
CheetahMask = zeros(size(imay_pad,1),size(imay_pad,2));
current_blocks = zeros(1,64);
DCT_blocks = zeros(size(imaC_pad,1),size(imaC_pad,2),64);
CheetahMask = zeros(size(imaC_pad,1),size(imaC_pad,2));
P_BG = size(TrainsampleDCT_BG,1)/(size(TrainsampleDCT_BG,1)+size(TrainsampleDCT_FG,1));
P_FG = 1 - P_BG;
D = [1 2 4 8 16 24 32 40 48 56 64];
%% DCT Blocks
for k = 1 : size(imaC_pad,1)-7
    for j = 1 : size(imaC_pad,2)-7
        current_dc2_blocks = dct2(imaC_pad(k:k+7,j:j+7));
        current_blocks(Path) = current_dc2_blocks;
        DCT_blocks(k,j,:) = current_blocks(:);
    end
end
%% Initialization
MUFG = cell(5,1);
MUBG = cell(5,1);
SIgmaFG = cell(5,1);
SIgmaBG = cell(5,1);
PRiorFG = cell(5,1);
PRiorBG = cell(5,1);
%% P(a) FG
for i = 1:5
    [mu_i, sigma_i, prior_i] = initialEM(8);
    [MUFG{i},SIgmaFG{i},PRiorFG{i}] = EM(TrainsampleDCT_FG(:,:),8,64,mu_i(:,:),sigma_i(:,:,:),prior_i);
end

%% BG
for i = 1:5
    [mu_i, sigma_i, prior_i] = initialEM(8);
    [MUBG{i},SIgmaBG{i},PRiorBG{i}] = EM(TrainsampleDCT_BG(:,:),8,64,mu_i(:,:),sigma_i(:,:,:),prior_i);
end
%% P(b) Class
class = [1 2 4 8 16 32];
cMUFG = cell(6,1);
cMUBG = cell(6,1);
cSIgmaFG = cell(6,1);
cSIgmaBG = cell(6,1);
cPRiorFG = cell(6,1);
cPRiorBG = cell(6,1);
%%
for i = 1:6
    c = class(i);
    [mu_i, sigma_i, prior_i] = initialEM(c);
    [cMUFG{i},cSIgmaFG{i},cPRiorFG{i}] = EM(TrainsampleDCT_FG(:,:),c,64,mu_i(:,:),sigma_i(:,:,:),prior_i);
    [cMUBG{i},cSIgmaBG{i},cPRiorBG{i}] = EM(TrainsampleDCT_BG(:,:),c,64,mu_i(:,:),sigma_i(:,:,:),prior_i);
end
%% Classification P(a)
maskerror = zeros(25,11);
count = 1;
for mix1 = 1:5
    for mix2 = 1:5
        for d = 1:11
            priorfg = PRiorFG{mix1};
            priorbg = PRiorBG{mix2};
            mubg = MUBG{mix1};
            mufg = MUFG{mix2};
            sigmabg = SIgmaBG{mix1};
            sigmafg = SIgmaFG{mix2};
            for k = 1 : size(imaC_pad,1)-7
                for j = 1 : size(imaC_pad,2)-7
                    Pmix_G = 0;
                    Pmix_C = 0;
                    for c = 1:8
                        dct = DCT_blocks(k,j,1:D(d));
                        dct = squeeze(dct);
                        Grass = mymvn(dct',mubg(c,1:D(d)),sigmabg(1:D(d),1:D(d),c));
                        Pmix_G = Pmix_G + Grass*priorbg(c);
                        Cheetah = mymvn(dct',mufg(c,1:D(d)),sigmafg(1:D(d),1:D(d),c));
                        Pmix_C = Pmix_C + Cheetah*priorfg(c);
                    end
                    if (Pmix_C * P_FG) > (Pmix_G * P_BG)
                        CheetahMask(k,j) = 1;
                    else
                        CheetahMask(k,j) = 0;
                    end
                end
            end
            maskerror(count,d) = error(CheetahMask,imay_pad);
        end
        count = count + 1;
    end
end

%% Classification P(b)
cmaskerror = zeros(6,11);
count = 1;
for i = 1:6
    priorfg = cPRiorFG{i,1};
    priorbg = cPRiorBG{i,1};
    mubg = cMUBG{i,1};
    mufg = cMUFG{i,1};
    sigmabg = cSIgmaBG{i,1};
    sigmafg = cSIgmaFG{i,1};
    for d = 1:11
        for k = 1 : size(imaC_pad,1)-7
            for j = 1 : size(imaC_pad,2)-7
                Pmix_G = 0;
                Pmix_C = 0;
                for c = 1 : class(i)
                    dct = DCT_blocks(k,j,1:D(d));
                    dct = squeeze(dct);
                    Grass = mymvn(dct',mubg(c,1:D(d)),sigmabg(1:D(d),1:D(d),c));
                    Pmix_G = Pmix_G + Grass*priorbg(c);
                    Cheetah = mymvn(dct',mufg(c,1:D(d)),sigmafg(1:D(d),1:D(d),c));
                    Pmix_C = Pmix_C + Cheetah*priorfg(c);
                end
                if (Pmix_C * P_FG) > (Pmix_G * P_BG)
                    CheetahMask(k,j) = 1;
                else
                    CheetahMask(k,j) = 0;
                end
            end
        end
        cmaskerror(count,d) = error(CheetahMask,imay_pad);
    end
    count = count + 1;
end

%% Plot figure p(a)
f1 = figure('Renderer', 'painters', 'Position', [10 10 900 600]);

hold on
for i = 21:25
    plot(D,maskerror(i,:),"-o");
end

title('POE and Dimemsion with different parameters mixture5')
xlabel('Dimensions');
ylabel('probability of error');
label = legend;
set(label, 'location','northeastoutside');
hold off;
saveas(f1,'a_5.fig');
saveas(f1,'a_5.png');
saveas(f1,'a_5.jpg');
%% Plot figure p(b)
f2 = figure('Renderer', 'painters', 'Position', [10 10 900 600]);

hold on
for i = 1:6
    plot(D,cmaskerror(i,:),"-o");
end

title('POE and Dimemsion with different components')
xlabel('Dimensions');
ylabel('probability of error');
label = legend({'C=1','C=2','C=4','C=8','C=16','C=32'});
set(label, 'location','northeastoutside');
hold off;
saveas(f2,'b.fig');
saveas(f2,'b.png');
saveas(f2,'b.jpg');
%% Function
function [mu,sigma,prior] = initialEM(class)
prior = rand(1,class);
prior(prior < 1e-4) = 1e-4;
prior = prior /sum(prior);
mu = rand(class,64);
mu(mu<1e-4) = 1e-4;
sigma = zeros(64,64,class);
s_r = rand(class,64);
s_r(s_r < 1e-4) = 1e-4;
for i = 1:class
    sigma(:,:,i) = diag(s_r(i,:));
end
end

function [mu,sigma,p] = EM(x,class,~,mu,sigma,prior)
mu_pr = mu;
sigma_pr = sigma;
joint_p = zeros(size(x,1),class);
for i = 1 : 200
    for k = 1: size(x,1)
        for j = 1: class
            joint_p(k,j) = mymvn(x(k,:),mu(j,:),sigma(:,:,j))* prior(j);
        end
    end
    hij = joint_p./repmat(sum(joint_p,2),1,class);
    hij(hij<1e-6) = 1e-6;
    for j = 1:class
        mu(j,:) = sum(hij(:,j).*x)./sum(hij(:,j));
        prior(1,j) = sum(hij(:,j))/ size(x,1);
        temp = (x-mu(j,:))'* diag(hij(:,j)) * (x-mu(j,:)) ./ sum(hij(:,j));
        temp(temp<1e-4) = 1e-4;
        sigma(:,:,j) = diag(diag(temp)); % Could be better, though
    end
    mu_pr = mu;
    sigma_pr = sigma;
    p = sum(hij)/size(x,1);
end
end


function error = error(pred, true)
error = sum(abs(pred-true),'all')/(256*272);
end

function y = mymvn(x,mean,cov)
constant = sqrt(det(cov)*(2*pi)^(size(cov,2)));
exp_component = -0.5 * (x-mean) /(cov) * (x-mean)';
y = exp(exp_component)/constant;

end
