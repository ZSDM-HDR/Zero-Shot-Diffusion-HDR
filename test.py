# inference

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from cldm.ddim_hacked_my_encdec import DDIMSampler
import torch
import os


#=================================params
parser = argparse.ArgumentParser()
parser.add_argument('-input_path', help='the folder of MSCN maps', default='./test_imgs/', type=str)  
parser.add_argument('-output_path', help='the folder for output images', default='./test_imgs_output/', type=str)
parser.add_argument('-model_yaml_path', help='the path of the model configuration file', default='./models/cldm_v15_inference.yaml', type=str)
parser.add_argument('-model_weight_path', help='the path of the model weight', default='./model_weights/last.ckpt', type=str)
parser.add_argument('-ddim_steps', default=20, type=int)    #recommend: 20
parser.add_argument('-random_seed', default=2, type=int)
parser.add_argument('-guess_mode', default=False, type=bool)
parser.add_argument('-scale', help='classifier-free guidance hyperparam', default=9.0, type=float)
parser.add_argument('-strength', default=1.0, type=float)
parser.add_argument('-post_processing', help='0: no post-processing, 1: simple processing; 2: smoothed gain ratio', default=1, type=int)   #2 is recommend for the LVZHDR dataset.
parser.add_argument('-norm_hdr_first', help='whether to normalize the hdr image before processing', default=True, type=bool)    #The impact of different norm strategies is quite minimal.
#====================================
args = parser.parse_args()

#input sources
prompt = ''
a_prompt = ''
n_prompt = ''
num_samples = 1     #how many images will be generated
image_resolution = 640  
ddim_steps = args.ddim_steps
guess_mode = args.guess_mode
strength = args.strength  
scale = args.scale
seed = args.random_seed
eta = 0.0

#instantiate the model
model = create_model(args.model_yaml_path).cpu()    #config for inference
model.load_state_dict(load_state_dict(args.model_weight_path, location='cuda'),strict=False)   #best
model = model.cuda()
ddim_sampler = DDIMSampler(model)

#================ multi scale mscn and luma input
def process(ori_mscn, mscn_map, luma_map, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
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
        samples, intermediates = ddim_sampler.sample(ori_mscn, ddim_steps, num_samples,       
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
def test_big_size(ori_mscn, mscn_map, luma_map, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, patch_h = 640, patch_w = 640, patch_h_overlap = 320, patch_w_overlap = 320):

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
            with torch.no_grad():
                test_patch_result = process(ori_mscn[h_begin:h_end,w_begin:w_end,:], test_mscn_patch, test_luma_patch, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
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
        with torch.no_grad():
            test_patch_result = process(ori_mscn[h_begin:h_end,-patch_w:,:]   , test_mscn_patch, test_luma_patch, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)    
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
        with torch.no_grad():
            test_patch_result = process(ori_mscn[-patch_h:,w_begin:w_end,:], test_mscn_patch, test_luma_patch, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
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
    with torch.no_grad():
        test_patch_result = process(ori_mscn[-patch_h:,-patch_w:,:], test_mscn_patch, test_luma_patch, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
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
    #last_last_range = patch_h
    for i in range(last_last_range):
        weight1 = np.float32((last_last_range-1-i)/(last_last_range-1))
        weight2 = np.float32(i/(last_last_range-1))
        test_result[H-patch_h+i,:,:] = test_result[H-patch_h+i,:,:]*weight1 + test_horizontal_result[i,:,:]*weight2
    test_result[h_end:,:,:] = test_horizontal_result[last_last_range:,:,:]
   
    return test_result


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

mscn_names = os.listdir(args.input_path + 'test_mscn_input/')
for mscn_name in mscn_names:
    mscn_map = cv2.imread(args.input_path + 'test_mscn_input/' + mscn_name)
    luma_map = cv2.imread(args.input_path + 'test_luma_input/' + mscn_name)    
    ori_img  = cv2.imread(args.input_path + 'test_orihdr_input/' + mscn_name.replace('png', 'hdr'), flags = cv2.IMREAD_ANYDEPTH)
    if args.norm_hdr_first:
        ori_img = (ori_img - np.min(ori_img)) / (np.max(ori_img) - np.min(ori_img))
        ori_img = cv2.cvtColor(np.float32(ori_img), cv2.COLOR_BGR2RGB)
    ori_mscn,ori_mscn_norm,_,_ = mscn(True, ori_img, 7, 0.0000001)  
    
    result = test_big_size(ori_mscn, mscn_map, luma_map, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
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