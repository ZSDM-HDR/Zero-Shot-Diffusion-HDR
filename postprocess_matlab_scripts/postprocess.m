clear;
eps_ldr = 0.0000001;
eps_hdr = 0.0000001;
ksize_ldr = 7;
ksize_hdr = 7;
s = 0.5;
enhance = 1;
smooth_ratio=0;

LDR_Path = 'D:\matlab code\cvpr2024_hdr\test_imgs\predicted\';   
HDR_Path = 'D:\matlab code\cvpr2024_hdr\test_imgs\ori_hdr\';
output_Path = 'test/';

File = dir(fullfile(LDR_Path,'*.png'));  
FileNames = {File.name}';            
length = size(FileNames,1);

window = fspecial('gaussian',7,7/6);
window = window/sum(sum(window));

for i=1:length
    pred_img_path = strcat(LDR_Path,FileNames{i});
    pred_img = double(imread(pred_img_path)) / 255;
    [h,w,c] = size(pred_img);
    [pred_mscn, pred_mu, pred_sigma] = MSCN(pred_img, ksize_ldr, eps_ldr);
    
    ori_img_path = strcat(HDR_Path, strrep(FileNames{i}, 'png','hdr'));    %for HDR
    ori_img = double(hdrread(ori_img_path));
    ori_img = (ori_img - min(ori_img(:))) / (max(ori_img(:)) - min(ori_img(:)));
    ori_img = imresize(ori_img, [h,w], 'bilinear');
    [ori_mscn, ~, ~] = MSCN(ori_img, ksize_hdr, eps_hdr);
    
    pred_Y = rgb2gray(pred_img);
    ori_Y = rgb2gray(ori_img);
    
    tar_img_Y = ori_mscn.*(enhance*pred_sigma+eps_ldr) + pred_mu;

    if smooth_ratio==1
        ratio = tar_img_Y ./ ori_Y;
        ratio_smoothed = filter2(window, ratio, 'same');
        tar_img_Y = ori_Y .* ratio_smoothed;
    end

    tar_img = zeros(h,w,c);
    tar_img(:,:,1) = (ori_img(:,:,1) ./ (ori_Y+0.0000001)) .^s .* tar_img_Y;
    tar_img(:,:,2) = (ori_img(:,:,2) ./ (ori_Y+0.0000001)) .^s .* tar_img_Y;
    tar_img(:,:,3) = (ori_img(:,:,3) ./ (ori_Y+0.0000001)) .^s .* tar_img_Y;

    
    imwrite(tar_img, strcat(output_Path, FileNames{i}));
    FileNames{i}
end