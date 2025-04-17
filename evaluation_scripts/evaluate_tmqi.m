%hdr_list = dir(['hdr_survey_dataset\','*.hdr']);
ldr_list = dir(['.\results\','*.png']);
num = length(ldr_list);

Q = zeros(num,1);
S = zeros(num,1);
N = zeros(num,1);

for i = 1:num
    ldr_path = ['results\',ldr_list(i).name];
    ldr = imread(ldr_path);
    [h,w,c] = size(ldr);

    hdr_path = ['hdr_survey_dataset\',strrep(ldr_name , "png", "hdr")];
    hdr = hdrread(hdr_path);
    hdr = imresize(hdr,[h,w],'bilinear');   

    % ============ TMQI
    [Q(i,1), S(i,1), N(i,1), s_maps, s_local] = TMQI(hdr, ldr);
    
end

xlswrite('tmqi_result.xlsx',{'Q','S','N','meanQ','meanS','meanN'},'sheet1','A1');
excel = [Q,S,N];
xlswrite('tmqi_result.xlsx',excel,'sheet1','A2');
meanQ = mean(Q);
meanS = mean(S);
meanN = mean(N);
xlswrite('tmqi_result.xlsx',meanQ,'sheet1','D2');
xlswrite('tmqi_result.xlsx',meanS,'sheet1','E2');
xlswrite('tmqi_result.xlsx',meanN,'sheet1','F2');
