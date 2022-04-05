import random
import numpy as np
import pickle
import time
import PIL
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import math 




class deepKp(Dataset):

    def __init__(self, data, target, transform=None):
        
        self.data = data
        self.transform = transform
        self.target = target

    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        
        
	    sample = torch.tensor(data[idx])
	    class_number = torch.tensor(int(target[idx]))
	    

	    if self.transform:
	        sample = self.transform(sample)

	    return (sample, class_number)


