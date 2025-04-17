niqe_val = 0;
file_path = '.\results\';
img_path_list = dir(strcat(file_path, '*.png'));
img_num = length(img_path_list);
for j = 1:img_num
    img_name = [file_path img_path_list(j).name];
    img = imread(img_name);
    disp(img_path_list(j).name);
    cur_niqe_val = niqe(img);
    niqe_val = niqe_val + cur_niqe_val;
    %fprintf('niqe val: %f \n', cur_niqe_val);
end
niqe_val = niqe_val / img_num;
fprintf('Average NIQE value: %f \n', niqe_val);
