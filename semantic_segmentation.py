import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image

class VOCDataSet(torch.utils.data.Dataset):
    def __init__(self, root, list_path, ignore_label=255):
        super(VOCDataSet,self).__init__()
        self.root = root
        self.list_path = list_path
		self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(self.root, "UAVSegImages/%s.jpg" % name)
            label_file = os.path.join(self.root, "UAVSegLabels/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file
            })
 
    def __len__(self):
        return len(self.files)
 
 
    def __getitem__(self, index):
        datafiles = self.files[index]
 
        '''load the datas'''
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"]).convert('L')
        size_origin = image.size # W * H

		I = np.asarray(image,np.float32) 
        image = I.transpose((2,0,1))#transpose the  H*W*C to C*H*W
        label = np.asarray(np.array(label), np.int64)
        return image, label

