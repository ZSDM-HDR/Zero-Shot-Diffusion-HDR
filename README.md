# Zero-Shot-Diffusion-HDR
This is a PyTorch implementation of [CVPR 2024] paper: [Zero-Shot Structure-Preserving Diffusion Model for High Dynamic Range Tone Mapping](https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Zero-Shot_Structure-Preserving_Diffusion_Model_for_High_Dynamic_Range_Tone_Mapping_CVPR_2024_paper.html).

*__Notice__: This repo is still __working in progress__. We are also currently extending this conference paper into a journal version. Due to our affiliation's regulations, we will apply to publish the codes for more functions and the model weights after the extended version is completed. If you would now like to use our model in your research, please contact the authors by email.*

## Environment Set-Up
Please follow the instructions of [ControlNet](https://github.com/lllyasviel/ControlNet) to set up the environment.

## Prepare Datasets and Pre-Trained Models
In our paper, we use the high-resolution part of [Flickr2K](https://github.com/limbee/NTIRE2017) dataset as the training set. To prepare the training data, you should first generate the MSCN maps and blurred luma maps following the instructions in our paper, and put them in `training/Flickr2K/source/` and `training/Flickr2K/source1/` respectively. The original images should be put in `training/Flickr2K/target/`.

Then, download a pre-trained model ([StableDiffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) or [ControlNet](https://github.com/lllyasviel/ControlNet)), and put it in `my_model_weights/`, then change `resume_path` in `train.py`.

## Train
`python train.py`

To train the basic model, use `models/cldm_v15.yaml` as the config file.

To fine tune the model with decoding-encoding process, use `models/cldm_v15_ftenc.yaml` as the config file.

## Test
Put the MSCN maps, the blured luma maps, and the original images in `test_imgs/` as the example test image does. Use `models/cldm_v15_inference.yaml` as the config file.

`python test.py`

The post-processing introduced in our paper should be applied to the generated image to finnaly produce accurate color.


## Acknowledgement

Our implementation is based on this repository: [ControlNet](https://github.com/lllyasviel/ControlNet).
