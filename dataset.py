import json
import cv2
import numpy as np
import random

from torch.utils.data import Dataset


class MyDataset(Dataset):   #input MSCN and luma respectively
    def __init__(self):
        self.data = []
        with open('./training/Flickr2K/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']    #mscn
        source1_filename = source_filename.replace('source', 'source1') #luma
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/Flickr2K/' + source_filename)       #mscn
        source1= cv2.imread('./training/Flickr2K/' + source1_filename)      #luma
        target = cv2.imread('./training/Flickr2K/' + target_filename)

        #random crop
        w, h, c = source.shape
        x = random.randint(0,w-641)
        y = random.randint(0,h-641)
        source = source[x:x+640, y:y+640, :]            #MSCN
        source1 = source1[x:x+640, y:y+640, :]          #luma
        target = target[x:x+640, y:y+640, :]

        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # # Normalize source images to [0, 1], and cat.
        source = source.astype(np.float32) / 255.0
        source1 = source1.astype(np.float32) / 255.0


        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, luma_hint=source1)   