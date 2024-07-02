from share import *
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
    every_n_train_steps = 150,
    save_last = True,
    save_weights_only = False,
    filename = 'exp_sd_{epoch:02d}_{step:06d}'
)


# Configs
resume_path = './model_weights/last.ckpt'
batch_size = 8
logger_freq = 500
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
#load model
ControlNet = load_state_dict(resume_path, location='cpu')
model_dict = model.state_dict()
pretrained_dict = {k:v for k,v in ControlNet.items() if k in model_dict and (v.shape == model_dict[k].shape)}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict, strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=16, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(precision=32, callbacks=[logger, checkpoint_callback], max_epochs=200)


# Train!
trainer.fit(model, dataloader)
