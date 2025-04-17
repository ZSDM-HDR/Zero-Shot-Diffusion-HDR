function [Q, S, N, s_maps, s_local] = TMQI(hdrImage, ldrImage, window)
% ========================================================================
% Tone Mapped image Quality Index (TMQI), Version 1.0
% Copyright(c) 2012 Hojatollah Yeganeh and Zhou Wang
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is hereby
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
% This is an implementation of an objective image quality assessment model
% for tone mapped low dynamic range (LDR) images using their corresponding
% high dynamic range (HDR) images as references.
% 
% Please refer to the following paper and the website with suggested usage
%
% H. Yeganeh and Z. Wang, "Objective Quality Assessment of Tone Mapped
% Images," IEEE Transactios on Image Processing, vol. 22, no. 2, pp. 657- 
% 667, Feb. 2013.
%
% http://www.ece.uwaterloo.ca/~z70wang/research/tmqi/
%
% Kindly report any suggestions or corrections to hyeganeh@ieee.org,
% hojat.yeganeh@gmail.com, or zhouwang@ieee.org
%
%----------------------------------------------------------------------
%
%Input : (1) hdrImage: the HDR image being used as reference. The HDR 
%            image must be an m-by-n-by-3 single or double array
%        (2) ldrImage: the LDR image being compared
%        (3) window: local window for statistics (see the above
%            reference). default widnow is Gaussian given by
%            window = fspecial('gaussian', 11, 1.5);
%
%Output: (1) s_maps: The structural fidelity maps of the LDR image.     
%        (2) s_local: the mean of s_map in each scale (see above refernce).
%        (3) S: The structural fidelity score of the LDR test image. 
%        (4) N: The statistical naturalness score of the LDR image.
%        (5) Q: The TMQI score of the LDR image. 
%
%Basic Usage:
%   Given LDR test image and its corresponding HDR images, 
%
%   [Q, S, N, s_maps, s_local] = TMQI(hdrImage, ldrImage);
%
%Advanced Usage:
%   User defined parameters. For example
%   window = ones(8);
%   [Q, S, N, s_maps, s_local] = TMQI(hdrImage, ldrImage, window);
%
%========================================================================

if (nargin < 2 || nargin > 3)
   s_maps = -Inf;
   s_local = -Inf;
   S = -Inf;
   N = -Inf;
   Q = -Inf;
   return;
end

if (size(hdrImage) ~= size(ldrImage))
   s_maps = -Inf;
   s_local = -Inf;
   S = -Inf;
   N = -Inf;
   Q = -Inf;
   return;
end

[M N D] = size(hdrImage);

if (nargin == 2)
   if ((M < 11) || (N < 11))
	   s_maps = -Inf;
       s_local = -Inf;
       S = -Inf;
       N = -Inf;
       Q = -Inf;
       disp('the image size is less than the window size'); 
     return
   end
   window = fspecial('gaussian', 11, 1.5);	%
end

if (nargin == 3)
   [H W] = size(window);
   if ((H*W) < 4 || (H > M) || (W > N))
	   s_maps = -Inf;
       s_local = -Inf;
       S = -Inf;
       N = -Inf;
       Q = -Inf;
      return
   end
end

%---------- default parameters -----
a = 0.8012;
Alpha = 0.3046;
Beta = 0.7088;
%---------- default parameters -----
level = 5;
weight = [0.0448 0.2856 0.3001 0.2363 0.1333];
%--------------------
HDR = RGBtoYxy(hdrImage);
L_hdr = HDR(:,:,1);
lmin = min(min(L_hdr));
lmax = max(max(L_hdr));
L_hdr = double(round((2^32 - 1)/(lmax - lmin)).*(L_hdr - lmin));
%-------------------------------------------
L_ldr = RGBtoYxy (double(ldrImage));
L_ldr = double(L_ldr(:,:,1));
%----------- structural fidelity -----------------
[S s_local s_maps] = StructuralFidelity(L_hdr, L_ldr,level,weight, window);
%--------- statistical naturalness ---------------
N = StatisticalNaturalness(L_ldr);
%------------- overall quality -----------------
Q =  a*(S^Alpha) + (1-a)*(N^Beta);
end

%================ Color Space Conversion ==============================

function [Yxy] = RGBtoYxy(RGB)
RGB = double(RGB);
M = [ 0.4124 0.3576 0.1805
      0.2126 0.7152 0.0722
      0.0193  0.1192 0.9505];

 RGB2 = reshape(RGB,size(RGB,1)*size(RGB,2),3);
 XYZ2 = M *  RGB2';
 S = sum(XYZ2);
 Yxy2 = zeros(size(XYZ2));
 Yxy2(1,:)=XYZ2(2,:);
 Yxy2(2,:)=XYZ2(1,:)./(S+eps);
 Yxy2(3,:)=XYZ2(2,:)./(S+eps);
 Yxy = reshape(Yxy2',size(RGB,1),size(RGB,2),3);
end

%================= Structural Fidelity Measure =========================

function [S s_local s_maps] = StructuralFidelity(L_hdr, L_ldr,level,weight, window)
downsample_filter = ones(2)./4;
L_hdr = double(L_hdr);
L_ldr = double(L_ldr);
f = 32;
for l = 1:level
    f = f/2;
    [s_local(l) smap] = Slocal(L_hdr, L_ldr, window , f);
    s_maps{l} = smap;
    filtered_im1 = imfilter(L_hdr, downsample_filter, 'symmetric', 'same');
    filtered_im2 = imfilter(L_ldr, downsample_filter, 'symmetric', 'same');
    clear L_hdr;
    clear L_ldr;
    L_hdr = filtered_im1(1:2:end, 1:2:end);
    L_ldr = filtered_im2(1:2:end, 1:2:end);
end
S = prod(s_local.^weight);
end

%============ Local Structural Fidelity Measure =========================

function [s, s_map] = Slocal(L_hdr, L_ldr,window, sf)
C1 = 0.01;
C2 = 10;
window = window/sum(sum(window));
%----------------------------------------------------
img1 = double(L_hdr);
img2 = double(L_ldr);
mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma1 = sqrt(max(0, sigma1_sq));
sigma2 = sqrt(max(0, sigma2_sq));
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;
%-------------------------------------------------------
% Mannos CSF Function
%f=8;
CSF = 100.*2.6*(0.0192+0.114*sf)*exp(-(0.114*sf)^1.1);
%-----------------------
u_hdr= 128/(1.4*CSF); 
sig_hdr=u_hdr/3;
sigma1p = normcdf(sigma1,u_hdr,sig_hdr);
%sigma1p(sigma1 < (u_hdr - 2*sig_hdr))=0;
%sigma1p(sigma1 > (u_hdr + 2*sig_hdr))=1;
%---------------------------------------------------------
u_ldr = u_hdr;
sig_ldr = u_ldr/3;
sigma2p = normcdf(sigma2,u_ldr,sig_ldr);
%sigma2p(sigma2< (u_ldr - 2*sig_ldr)) = 0;
%sigma2p(sigma2> (u_ldr + 2*sig_ldr)) = 1;
%----------------------------------------------------
s_map = (((2*sigma1p.*sigma2p)+C1)./((sigma1p.*sigma1p)+(sigma2p.*sigma2p)+C1)).*((sigma12+C2)./(sigma1.*sigma2 + C2));
s = mean2(s_map);
end

%============ Statistical Naturalness Measure ================

function [N] = StatisticalNaturalness(L_ldr)
u = mean2(L_ldr);
fun = @(x) std(x(:))*ones(size(x));
I1 = blkproc(L_ldr,[11 11],fun);
sig = (mean2(I1));
%------------------ Contrast ----------
phat(1) = 4.4;
phat(2) = 10.1;
beta_mode = (phat(1) - 1)/(phat(1) + phat(2) - 2);
C_0 = betapdf(beta_mode, phat(1),phat(2));
C   = betapdf(sig./64.29,phat(1),phat(2));
pc = C./C_0;
%----------------  Brightness ---------
muhat = 115.94;
sigmahat = 27.99;
B = normpdf(u,muhat,sigmahat);
B_0 = normpdf(muhat,muhat,sigmahat);
pb = B./B_0;
%-------------------------------
N = pb*pc;
end

%========================================================================