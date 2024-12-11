clear

%==========get file names
Path = 'D:/matlab_code/gen_luma_and_mscn/hdr_dataset/';                  
File = dir(fullfile(Path,'*.hdr'));  
FileNames = {File.name}';            

%==========get lambdas
img_names = {'name'};
fpn = fopen ('D:/matlab_code/gen_luma_and_mscn/img_names.txt', 'rt');          
while feof(fpn) ~= 1                 
      file = fgetl(fpn);
      img_names = cat(1,img_names,file);
end
img_names = img_names(2:end);
lambda_values = importdata('D:/matlab_code/gen_luma_and_mscn/lambda_values.txt');
lambda_struct.name = 0;
length_npy = size(img_names,1);
M = [fieldnames(lambda_struct)'; struct2cell(lambda_struct)'];
for i = 1 : length_npy
    name = img_names(i);
    lambda = lambda_values(i);
    temp_struct = struct(cell2mat(name), lambda);
    M = cat(2,M,[fieldnames(temp_struct)'; struct2cell(temp_struct)']);
end
lambda_struct = struct(M{:});

%=============gen luma and mscn according to lambdas
Length_Names = size(FileNames,1);
for k = 1 : Length_Names
    file_name = FileNames(k);
    file_name = cell2mat(file_name);
    file_name_ = file_name(1:end-4);
    lambda = 25.5 * lambda_struct.(file_name_);
    hdr = double(hdrread(strcat('D:/matlab_code/gen_luma_and_mscn/hdr_dataset/', file_name))); 
    %============================resize (960x...)
    [h,w,c] = size(hdr);
    if h>w
        hdr = imresize(hdr, 960/w,'bilinear');  
    else
        hdr = imresize(hdr, 960/h,'bilinear');
    end
    hdr = hdr / max(hdr(:));                
    hdr = rgb2gray(hdr);
    
    %simple tone mapping
    hdr_tm = log(lambda * hdr / max(hdr(:)) + 0.000001) / log(lambda + 0.000001);
    
    %low pass luma map
    [h1,w1] = size(hdr_tm);
    window = fspecial('average',round(max([h1,w1])*0.02));
    window = window/sum(sum(window));
    lp = filter2(window, hdr_tm, 'same');
    
    %cal mscn
    [mscn_hdr, mu_hdr, sigma_hdr] = MSCN(hdr, 7, 0.0000001);
    mscn_hdr_norm = (mscn_hdr - min(mscn_hdr(:))) / (max(mscn_hdr(:)) - min(mscn_hdr(:)));  
    
    %save
    imwrite(mscn_hdr_norm, strcat('mscn_norm/', file_name_, '.png'));
    imwrite(lp, strcat('luma/', file_name_, '.png'));
        
end

