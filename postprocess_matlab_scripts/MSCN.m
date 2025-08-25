function [struct, mu, sigma] = MSCN(input, ksize, c)
if ndims(input)==3
    input = double(rgb2gray(input));
end

window = fspecial('gaussian',ksize,ksize/6);
window = window/sum(sum(window));

mu      = filter2(window, input, 'same');
mu_sq   = mu.*mu;
sigma   = sqrt(abs(filter2(window, input.*input, 'same') - mu_sq));
struct  = (input-mu)./(sigma+c);
%struct     = (input-mu)./(sigma+0.0001);
end

