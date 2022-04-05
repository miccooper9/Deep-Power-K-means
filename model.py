import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import math


class PowerKdeep(nn.Module):

	def __init__(self, enc_in2, enc_in3, emb_in, emb_out, n_clusters, inp = 784):



		super(PowerKdeep, self).__init__()

	    #encoder
	    self.enc1= torch.nn.Linear(inp, enc_in2)
	    self.enc2 = torch.nn.Linear(enc_in2, enc_in3)
	    self.enc3 = torch.nn.Linear(enc_in3, emb_in)
	    self.encout = torch.nn.Linear(emb_in, emb_out)

	    #decoder
	    self.dec1= torch.nn.Linear(emb_out, emb_in)
	    self.dec2 = torch.nn.Linear(emb_in, enc_in3)
	    self.dec3 = torch.nn.Linear(enc_in3, enc_in2)
	    self.decout = torch.nn.Linear(enc_in2, inp)


	    self.cluster_reps = nn.Parameter(data=torch.randn(n_clusters, emb_out).uniform_(-1, 1), requires_grad=True)



	def forward(self, x):

		#encoder
	    h1_out = F.relu(self.enc1(x))
	    h2_out = F.relu(self.enc2(h1_out))
	    h3_out = F.relu(self.enc3(h2_out))
	    embedding = self.encout(h3_out)

	    #decoder
	    h5_out = F.relu(self.dec1(embedding))
	    h6_out = F.relu(self.dec2(h5_out))
	    h7_out = F.relu(self.dec3(h6_out))
	    out = self.decout(h7_out)

	    return embedding,self.cluster_reps,out




	    









  
