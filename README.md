# Zero-Shot-Diffusion-HDR
This is a PyTorch implementation of [CVPR 2024] paper: [Zero-Shot Structure-Preserving Diffusion Model for High Dynamic Range Tone Mapping](https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Zero-Shot_Structure-Preserving_Diffusion_Model_for_High_Dynamic_Range_Tone_Mapping_CVPR_2024_paper.html).

A journal version of this work, entitled [A Flexible Zero-Shot Approach to Tone Mapping via Structure-Preserving Diffusion Models](https://ieeexplore.ieee.org/document/11129103), is accepted by IEEE TCSVT.

## Environment Set-Up
Please follow the instructions of [ControlNet](https://github.com/lllyasviel/ControlNet) to set up the environment.

## Prepare Datasets and Pre-Trained Models
For training:

In our paper, we use the high-resolution part of [Flickr2K](https://github.com/limbee/NTIRE2017) dataset as the training set. To prepare the training data, you should first generate the MSCN maps and blurred luma maps following the instructions in our paper, and put them in `training/Flickr2K/source/` and `training/Flickr2K/source1/` respectively. The original images should be put in `training/Flickr2K/target/`.

Then, download a pre-trained model ([StableDiffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) or [ControlNet](https://github.com/lllyasviel/ControlNet)), and put it in `my_model_weights/`, then change `resume_path` in `train.py`.

For simple testing:

You can also download our pretrained model [here](https://drive.google.com/file/d/15YKrb-aBjDebN7E0JYMUctpwgRRfNmmm/view?usp=sharing).

## Train
`python train.py`

To train the basic model, use `models/cldm_v15.yaml` as the config file.

To fine tune the model with decoding-encoding process, use `models/cldm_v15_ftenc.yaml` as the config file.

## Test
Put the MSCN maps, the blured luma maps, and the original images in `test_imgs/` as the example test image does. Use `models/cldm_v15_inference.yaml` as the config file.

`python test.py`

If necessary, the parameters and arguments may be changed (directories, pre-processing and post-processing methods, etc.). Please follow the instructions in the code.

## Test the FTA strategy
To test the FTA strategy we introduced in our journal version paper, you can run the script `test_fta.py`:

`python test_fta.py`

If necessary, the parameters and arguments may be changed. Please follow the instructions in the code. If using the style matching loss, you need to download the pre-trained DA-CLIP-like encoder [here](https://drive.google.com/file/d/1sI6mq8ihTGNggc8BDZjV1Z6YXvY-R7lW/view?usp=sharing).

## Acknowledgement

Our implementation is based on this repository: [ControlNet](https://github.com/lllyasviel/ControlNet).
