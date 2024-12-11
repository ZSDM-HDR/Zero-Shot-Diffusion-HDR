function [struct, mu, sigma] = MSCN(input, ksize, c)
input = double(input);

window = fspecial('gaussian',ksize,ksize/6);
window = window/sum(sum(window));

mu      = filter2(window, input, 'same');
mu_sq   = mu.*mu;
sigma   = sqrt(abs(filter2(window, input.*input, 'same') - mu_sq));
struct  = (input-mu)./(sigma+c);
end

