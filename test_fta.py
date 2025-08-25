#test the FTA strategy
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from share import *
import config
import argparse

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from daclip_model.factory import create_model as create_daclip_model
from cldm.ddim_hacked_encdec_cond import DDIMSampler    
import torch
import os
import torch.nn.functional as F
from math import exp
from torchvision.transforms import Resize


#=================================params
parser = argparse.ArgumentParser()
parser.add_argument('-input_path', help='the folder of MSCN maps', default='./test_imgs/', type=str)  
parser.add_argument('-output_path', help='the folder for output images', default='./test_imgs_output/', type=str)
parser.add_argument('-model_yaml_path', help='the path of the model configuration file', default='./models/cldm_v15_inference.yaml', type=str)
parser.add_argument('-model_weight_path', help='the path of the model weight', default='./model_weights/last.ckpt', type=str)
parser.add_argument('-ddim_steps', default=20, type=int)
parser.add_argument('-random_seed', default=2, type=int)
parser.add_argument('-guess_mode', default=False, type=bool)
parser.add_argument('-scale', help='classifier-free guidance hyperparam', default=9.0, type=float)
parser.add_argument('-strength', default=1.0, type=float)
parser.add_argument('-post_processing', help='0: no post-processing, 1: simple processing; 2: smoothed gain ratio', default=1, type=int)   #2 is recommend for the LVZHDR dataset.
parser.add_argument('-norm_hdr_first', help='whether to normalize the hdr image before processing', default=True, type=bool)    #The impact of different norm strategies is quite minimal.
parser.add_argument('-ref_img_path', help='the reference image used for style matching loss', default=None, type=str)
parser.add_argument('-mask1_path', help='the mask used for the regional loss', default=None, type=str)  
parser.add_argument('-fta_strategy', help='1: z_{t-1} minus the gradient; 2: z_0 minus the gradient', default=1, type=int)
parser.add_argument('-brightness_loss_scale', help='the scale of brightness loss', default=0, type=float)   #recommend: 100~1000 if using brightness loss; else 0
parser.add_argument('-brightness_level', help='the expected brightness level (range: 0~1)', default=0, type=float)
parser.add_argument('-saliency_loss_scale', help='the scale of saliency loss', default=0, type=float)       #recommend: 100~1000 if using saliency loss; else 0
parser.add_argument('-style_loss_scale', help='the scale of style matching loss', default=0, type=float)    #recommend: 10000~50000 if using style matching loss; else 0
parser.add_argument('-daclip_model_pth', help='path of daclip model weight (if using the style matching loss)', default=None, type=str) 
#====================================
args = parser.parse_args()

#input sources
prompt = ''
a_prompt = ''
n_prompt = ''
num_samples = 1     #how many images will be  generated
image_resolution = 640  
ddim_steps = args.ddim_steps #20
guess_mode = args.guess_mode
strength = args.strength  
scale = args.scale
seed = args.random_seed
eta = 0.0


strategy = args.fta_strategy
brightness_loss_scale = args.brightness_loss_scale          #recommend: 100~1000 if using brightness loss; else 0
brightness_level = args.brightness_level                    #the expected brightness level
saliency_loss_scale = args.saliency_loss_scale              #recommend: 100~1000 if using saliency loss; else 0
style_guidance_scale = args.style_loss_scale                #recommend: 10000~50000 if using style matching loss; else 0
daclip_checkpoint = args.daclip_model_pth                   #for style matching loss
ref_img_path = args.ref_img_path                            #for style matching loss
mask1_path = args.mask1_path                                #for regional loss
norm_mean=torch.Tensor([0.48145466, 0.4578275, 0.40821073]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
norm_std=torch.Tensor([0.26862954, 0.26130258, 0.27577711]).unsqueeze(0).unsqueeze(2).unsqueeze(3)


#instantiate the model
model = create_model(args.model_yaml_path).cpu()    #config for inference
model.load_state_dict(load_state_dict(args.model_weight_path, location='cuda'),strict=False)   #best
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def gray_dgp(image):
    n = int(image.shape[0])
    h = int(image.shape[2])
    w = int(image.shape[3])
    r = image[:, 0, :, :]
    g = image[:, 1, :, :]
    b = image[:, 2, :, :]
    gray = 1/3 * r + 1/3 * g + 1/3 * b
    image_res = gray.view(n, 1, h, w).expand(n, 3, h, w)
    return image_res

def mscn(isrgb, input_img, ksize, c):  #input an RGB image
    if isrgb==True:
        yuv_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2YUV).astype(np.float32)
        y = np.expand_dims(yuv_img[:,:,0], axis=-1).astype(np.float64)
    else:
        y = input_img.astype(np.float64)
    mu = cv2.GaussianBlur(y, (ksize,ksize), ksize/6).astype(np.float64)
    mu_sq = mu * mu
    sigma = np.sqrt(np.absolute(cv2.GaussianBlur(y*y, (ksize,ksize), ksize/6) - mu_sq)).astype(np.float64)
    mu = np.expand_dims(mu, axis=-1)
    sigma = np.expand_dims(sigma, axis=-1)
    dividend = y.astype(np.float64) - mu
    divisor = sigma + c
    struct = dividend / divisor
    struct = struct.astype(np.float64)
    struct_norm = (struct - struct.min()) / (struct.max() - struct.min())
    return struct,struct_norm, mu, sigma

def gaussian(window_size, sigma):   # return a 1-d Gaussian vector given kernel size and sigma
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):  #return a 2-d Gaussian kernel
    _1D_window = gaussian(window_size, window_size/6.0).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# ============================================= define the loss functions
window = create_window(7)
def saliency_loss(input_img, window):
    if(input_img.shape[1]==3):
        input_img = torch.mean(input_img, dim=1).unsqueeze(1)
    pad = 3    #(7-1)/2
    window = window.to(input_img.device)
    gaussian_mu = F.conv2d(input_img, window, padding=pad, groups=1)   
    whole_mu = torch.mean(input_img)
    diff = torch.abs(gaussian_mu - whole_mu)
    return torch.mean(diff)

def brightness_loss(input_img, brightness_level):
    return (torch.mean(input_img)- brightness_level) ** 2
# ========================================================================


def process(fta_info, mask1_patch, ori_hdr, ori_luma, ref_style, StyleFilter, ori_mscn, mscn_map, luma_map, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    def hdrtm_cond_fn(x, t, model, daclip_model, ref_style):         
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)      #output of the diffusion model
            x_in_img = model.decode_first_stage(x_in)
            torch_resize = Resize([224,224])    
            x_in_img = torch_resize(x_in_img)
            x_in_img = x_in_img * 0.5 + 0.5
            x_in_img = torch.clamp(x_in_img, min=0, max=1)
            x_in_gray = x_in_img      
            x_in_gray = gray_dgp(x_in_gray)
            if (not ref_style is None) and (not daclip_model is None) and (style_guidance_scale != 0):
                daclip_out = daclip_model((x_in_gray - norm_mean) / norm_std)  
                img_feature = daclip_out["image_features"]
                mse_style = (img_feature - ref_style) ** 2
                mse_style = mse_style.mean(dim=(1))    
                mse_style = mse_style.sum()
            else:
                mse_style = 0
                
            brightness_loss_value = brightness_loss(x_in_gray, brightness_level)
            saliency_loss_value = saliency_loss(x_in_gray, window)

            loss = - brightness_loss_value * brightness_loss_scale 
            loss = loss + saliency_loss_scale * saliency_loss_value    
            loss = loss - mse_style * style_guidance_scale
            return torch.autograd.grad(loss, x_in)[0]   # take the gradient

    cond_fn = lambda x, t, model, daclip_model, ref_style: hdrtm_cond_fn(x, t, model, daclip_model, ref_style)
    with torch.no_grad():
        mscn_map = resize_image(mscn_map, image_resolution)
        luma_map = resize_image(luma_map, image_resolution)
        H, W, C = mscn_map.shape

        mscn_map = torch.from_numpy(mscn_map.copy()).float().cuda() / 255.0
        mscn_map = torch.stack([mscn_map for _ in range(num_samples)], dim=0)
        mscn_map = einops.rearrange(mscn_map, 'b h w c -> b c h w').clone()

        luma_map = torch.from_numpy(luma_map.copy()).float().cuda() / 255.0
        luma_map = torch.stack([luma_map for _ in range(num_samples)], dim=0)
        luma_map = einops.rearrange(luma_map, 'b h w c -> b c h w').clone()

        ori_luma = torch.from_numpy(ori_luma.copy()).float().cuda() / 255.0
        ori_luma = torch.stack([ori_luma for _ in range(num_samples)], dim=0)
        ori_luma = einops.rearrange(ori_luma, 'b h w c -> b c h w').clone()

        ori_hdr = torch.from_numpy(ori_hdr.copy()).float().cuda() / 255.0
        ori_hdr = torch.stack([ori_hdr for _ in range(num_samples)], dim=0)
        ori_hdr = einops.rearrange(ori_hdr, 'b h w c -> b c h w').clone()

        if mask1_patch is not None:
            mask1_patch = torch.from_numpy(mask1_patch.copy()).float().cuda()
            mask1_patch = torch.stack([mask1_patch for _ in range(num_samples)], dim=0)
            mask1_patch = einops.rearrange(mask1_patch, 'b h w c -> b c h w').clone()
            torch_resize = Resize([80,80]) 
            mask1_resized = torch_resize(mask1_patch)
            mask1_resized = mask1_resized[:,0,:,:].expand(1,4,80,80)
        else:
            mask1_resized = None

        fta_info_patch = dict(strategy=fta_info["strategy"], max_grad=fta_info["max_grad"], mask1=mask1_resized)

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
            

        cond = {"c_concat": [mscn_map], "c_concat_luma": [luma_map], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [mscn_map], "c_concat_luma": None if guess_mode else [luma_map], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(fta_info_patch, ori_hdr, ori_luma, ref_style, StyleFilter, cond_fn, ori_mscn, ddim_steps, num_samples,       
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)

        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results


#================ overlap processing
def test_big_size(fta_info, ori_hdr, ori_luma, ref_style, StyleFilter, ori_mscn, mscn_map, luma_map, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, patch_h = 640, patch_w = 640, patch_h_overlap = 320, patch_w_overlap = 320):

    H = mscn_map.shape[0]
    W = mscn_map.shape[1]
    print("H: ", H)
    print("W: ", W)
    
    test_result = np.zeros((H,W,3))
    h_index = 1
    while (patch_h*h_index-patch_h_overlap*(h_index-1)) < H:
        test_horizontal_result = np.zeros((patch_h,W,3))
        h_begin = patch_h*(h_index-1)-patch_h_overlap*(h_index-1)
        h_end = patch_h*h_index-patch_h_overlap*(h_index-1) 
        w_index = 1
        while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
            w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
            w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
            test_mscn_patch = mscn_map[h_begin:h_end,w_begin:w_end,:]  
            test_luma_patch = luma_map[h_begin:h_end,w_begin:w_end,:]
            test_ori_luma_patch = ori_luma[h_begin:h_end,w_begin:w_end,:]
            test_ori_hdr_patch = ori_hdr[h_begin:h_end,w_begin:w_end,:]
            test_mask_patch = fta_info["mask1"][h_begin:h_end,w_begin:w_end,:] if fta_info["mask1"] is not None else None                         
            with torch.no_grad():
                test_patch_result = process(fta_info, test_mask_patch, test_ori_hdr_patch, test_ori_luma_patch, ref_style, StyleFilter, ori_mscn[h_begin:h_end,w_begin:w_end,:], test_mscn_patch, test_luma_patch, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
                test_patch_result = test_patch_result[0]
            if w_index == 1:
                test_horizontal_result[:,w_begin:w_end,:] = test_patch_result
            else:
                for i in range(patch_w_overlap):
                    weight1 = np.float32((patch_w_overlap-1-i)/(patch_w_overlap-1))
                    weight2 = np.float32(i/(patch_w_overlap-1))
                    test_horizontal_result[:,w_begin+i,:] = test_horizontal_result[:,w_begin+i,:]*weight1 + test_patch_result[:,i,:]*weight2
                test_horizontal_result[:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,patch_w_overlap:,:]
            w_index += 1                   
    
        test_mscn_patch = mscn_map[h_begin:h_end,-patch_w:,:]   
        test_luma_patch = luma_map[h_begin:h_end,-patch_w:,:]      
        test_ori_luma_patch = ori_luma[h_begin:h_end,-patch_w:,:]
        test_ori_hdr_patch = ori_hdr[h_begin:h_end,-patch_w:,:]
        test_mask_patch = fta_info["mask1"][h_begin:h_end,-patch_w:,:] if fta_info["mask1"] is not None else None
        with torch.no_grad():
            test_patch_result = process(fta_info, test_mask_patch, test_ori_hdr_patch, test_ori_luma_patch, ref_style, StyleFilter,ori_mscn[h_begin:h_end,-patch_w:,:], test_mscn_patch, test_luma_patch, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)    
            test_patch_result = test_patch_result[0]
        last_range = w_end-(W-patch_w)     
        for i in range(last_range):
            weight1 = np.float32((last_range-1-i)/(last_range-1))
            weight2 = np.float32(i/(last_range-1))
            test_horizontal_result[:,W-patch_w+i,:] = test_horizontal_result[:,W-patch_w+i,:]*weight1 + test_patch_result[:,i,:]*weight2
        test_horizontal_result[:,w_end:,:] = test_patch_result[:,last_range:,:]    

        if h_index == 1:
            test_result[h_begin:h_end,:,:] = test_horizontal_result
        else:
            for i in range(patch_h_overlap):
                weight1 = np.float32((patch_h_overlap-1-i)/(patch_h_overlap-1))
                weight2 = np.float32(i/(patch_h_overlap-1))
                test_result[h_begin+i,:,:] = test_result[h_begin+i,:,:]*weight1 + test_horizontal_result[i,:,:]*weight2
            test_result[h_begin+patch_h_overlap:h_end,:,:] = test_horizontal_result[patch_h_overlap:,:,:] 
        h_index += 1

    test_horizontal_result = np.zeros((patch_h,W,3))
    w_index = 1
    while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
        w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
        w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
        test_mscn_patch = mscn_map[-patch_h:,w_begin:w_end,:] 
        test_luma_patch = luma_map[-patch_h:,w_begin:w_end,:]
        test_ori_luma_patch = ori_luma[-patch_h:,w_begin:w_end,:]
        test_ori_hdr_patch = ori_hdr[-patch_h:,w_begin:w_end,:]
        test_mask_patch = fta_info["mask1"][-patch_h:,w_begin:w_end,:] if fta_info["mask1"] is not None else None                           
        with torch.no_grad():
            test_patch_result = process(fta_info, test_mask_patch, test_ori_hdr_patch, test_ori_luma_patch, ref_style, StyleFilter, ori_mscn[-patch_h:,w_begin:w_end,:], test_mscn_patch, test_luma_patch, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
            test_patch_result = test_patch_result[0]

        if w_index == 1:
            test_horizontal_result[:,w_begin:w_end,:] = test_patch_result
        else:
            for i in range(patch_w_overlap):
                weight1 = np.float32((patch_w_overlap-1-i)/(patch_w_overlap-1))
                weight2 = np.float32(i/(patch_w_overlap-1))
                test_horizontal_result[:,w_begin+i,:] = test_horizontal_result[:,w_begin+i,:]*weight1 + test_patch_result[:,i,:]*weight2
            test_horizontal_result[:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,patch_w_overlap:,:]   
        w_index += 1

    test_mscn_patch = mscn_map[-patch_h:,-patch_w:,:]   
    test_luma_patch = luma_map[-patch_h:,-patch_w:,:]
    test_ori_luma_patch = ori_luma[-patch_h:,-patch_w:,:]
    test_ori_hdr_patch = ori_hdr[-patch_h:,-patch_w:,:]
    test_mask_patch = fta_info["mask1"][-patch_h:,-patch_w:,:] if fta_info["mask1"] is not None else None        
    with torch.no_grad():  
        test_patch_result = process(fta_info, test_mask_patch, test_ori_hdr_patch, test_ori_luma_patch, ref_style, StyleFilter, ori_mscn[-patch_h:,-patch_w:,:], test_mscn_patch, test_luma_patch, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
        test_patch_result = test_patch_result[0]
    if W>patch_w:
        last_range = w_end-(W-patch_w) 
    else:
        last_range = W 
        w_end = W          
    for i in range(last_range):
        weight1 = np.float32((last_range-1-i)/(last_range-1))
        weight2 = np.float32(i/(last_range-1))
        test_horizontal_result[:,W-patch_w+i,:] = test_horizontal_result[:,W-patch_w+i,:]*weight1 + test_patch_result[:,i,:]*weight2
    test_horizontal_result[:,w_end:,:] = test_patch_result[:,last_range:,:] 

    if H>patch_h:
        last_last_range = h_end-(H-patch_h)
    else:
        last_last_range = H    
        h_end = H 
    for i in range(last_last_range):
        weight1 = np.float32((last_last_range-1-i)/(last_last_range-1))
        weight2 = np.float32(i/(last_last_range-1))
        test_result[H-patch_h+i,:,:] = test_result[H-patch_h+i,:,:]*weight1 + test_horizontal_result[i,:,:]*weight2
    test_result[h_end:,:,:] = test_horizontal_result[last_last_range:,:,:]
   
    return test_result

def simple_post_processing(pred_img, ori_img, enhance, s, smooth_ratio, ksize):
    ori_yuv_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2YUV).astype(np.float32)
    ori_y = np.expand_dims(ori_yuv_img[:,:,0], axis=-1).astype(np.float32)
    ori_mscn,_,_,_ = mscn(True, ori_img, 7, 0.0000001)
    _, _, pred_mu, pred_sigma = mscn(True, pred_img.astype(np.float32), 7, 0.0000001)
    tar_img_Y = ori_mscn * (enhance*pred_sigma + 0.0000001) + pred_mu
    if smooth_ratio:
        ratio = (tar_img_Y/255) / ori_y
        smoothed_ratio = np.expand_dims(cv2.GaussianBlur(ratio, (ksize,ksize), ksize/6), axis=-1)
        tar_img_Y = ori_y * smoothed_ratio * 255
    tar_img = (ori_img / (ori_y + 0.0000001)) ** s * tar_img_Y
    tar_img = tar_img
    return tar_img

# ====================================== Beginning inference here ====================================== #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if (daclip_checkpoint is not None) and (ref_img_path is not None):
    # ============ init DACLIP model
    daclip_model = create_daclip_model(
            model_name="daclip_ViT-B-32",
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=None,
            force_image_size=None,
            pretrained_image=False,
            output_dict=True,
        )
    checkpoint = torch.load(daclip_checkpoint, map_location='cpu')
    daclip_model.load_state_dict(checkpoint["state_dict"])
    # ============ read input images
    ref_img = cv2.imread(ref_img_path)  
    ref_img = cv2.cvtColor(np.float32(ref_img), cv2.COLOR_BGR2RGB)       
    ref_img = cv2.resize(ref_img, (224, 224))
    ref_img = torch.from_numpy(ref_img.copy()).float().cuda() / 255.0
    ref_img = torch.stack([ref_img for _ in range(1)], dim=0)
    ref_img = einops.rearrange(ref_img, 'b h w c -> b c h w').clone()
    daclip_model = daclip_model.to(ref_img.device)
    norm_mean = norm_mean.to(ref_img.device)
    norm_std = norm_std.to(ref_img.device)
    ref_img = (ref_img - norm_mean) / norm_std      
    #========= style vector of the reference image
    daclip_out = daclip_model(ref_img)
    ref_img_feature = daclip_out["image_features"]
else:
    daclip_model = None
    ref_img_feature = None

mscn_names = ["UpheavalDome.png"]
for mscn_name in mscn_names:
        mscn_map = cv2.imread(args.input_path + 'test_mscn_input/' + mscn_name)   
        luma_map = cv2.imread(args.input_path + 'test_luma_input/' + mscn_name)   
        ori_luma = cv2.imread(args.input_path + 'test_luma_input/' + mscn_name)
        ori_img  = cv2.imread(args.input_path + 'test_orihdr_input/' + mscn_name.replace('png', 'hdr'), flags = cv2.IMREAD_ANYDEPTH)
        if args.norm_hdr_first:
            ori_img = (ori_img - np.min(ori_img)) / (np.max(ori_img) - np.min(ori_img))
            ori_img = cv2.cvtColor(np.float32(ori_img), cv2.COLOR_BGR2RGB)
        if mask1_path is not None:
            mask1 = cv2.imread(mask1_path) 
        else:
            mask1 = None

        ori_mscn,ori_mscn_norm,_,_ = mscn(True, ori_img, 7, 0.0000001)          
        fta_info = dict(strategy=strategy, max_grad=0.25, mask1=mask1)
        result = test_big_size(fta_info, ori_img, ori_luma, ref_img_feature, daclip_model, ori_mscn, mscn_map, luma_map, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
        if not args.norm_hdr_first:
            ori_img = (ori_img - np.min(ori_img)) / (np.max(ori_img) - np.min(ori_img))
            ori_img = cv2.cvtColor(np.float32(ori_img), cv2.COLOR_BGR2RGB)
        if args.post_processing==1:
            result = simple_post_processing(result, ori_img, 1, 0.5, False, 7)
        elif args.post_processing==2:
            result = simple_post_processing(result, ori_img, 1, 0.5, True, 7)
        result = cv2.cvtColor(np.float32(result), cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output_path + mscn_name, result)
    
        print("========== single image processing done. ==========")