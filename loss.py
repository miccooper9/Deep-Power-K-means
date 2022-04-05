import random
import numpy as np
import torch
import math 
from sklearn.datasets import fetch_openml 
from scipy.optimize import linear_sum_assignment



def AE_loss(output, target):

	losses = torch.sum((output-target)**2, 1)
	losses.register_hook(lambda x: print(x))  
	mean_batch_loss = torch.mean(losses)

	return mean_batch_loss
	


def PowerK_loss(embeddings, cluster_reps, Power_K):

	e = torch.repeat_interleave(embeddings, n_clusters, dim=0)
	ebr = torch.reshape(e, (batch_size, n_clusters, embedding_size))
	cbr = cluster_reps.repeat(batch_size,1,1)
	dist = torch.sum((ebr-cbr)**2,2)
	Kolmo_mean = torch.pow(torch.mean(torch.pow(dist,Power_K),1),1/Power_K)
	mean_batch_loss = torch.mean(Kolmo_mean)

	return mean_batch_loss


def cluster_acc(y_true, y_pred):
	"""
	Calculate clustering accuracy. Require scikit-learn installed.
	(Taken from https://github.com/XifengGuo/IDEC-toy/blob/master/DEC.py)
	# Arguments
		y: true labels, numpy.array with shape `(n_samples,)`
		y_pred: predicted labels, numpy.array with shape `(n_samples,)`
	# Return
		accuracy, in [0,1]
	"""
	y_true = y_true.astype(np.int64)
	assert y_pred.size == y_true.size
	D = max(y_pred.max(), y_true.max()) + 1
	w = np.zeros((D, D), dtype=np.int64)
	for i in range(y_pred.size):
		w[y_pred[i], y_true[i]] += 1
	ind = linear_sum_assignment(w.max() - w) # Optimal label mapping based on the Hungarian algorithm
	ind = np.asarray(ind)
	ind = np.transpose(ind)

	return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size




def AE_loss_mod(output, target):

	losses = torch.sum((output-target)**2, 1)
	losses.register_hook(lambda x: torch.clamp(x, min=-100, max=100))  
	mean_batch_loss = torch.mean(losses)

	return mean_batch_loss

def PowerK_loss_mod(embeddings, cluster_reps, Power_K):

	e = torch.repeat_interleave(embeddings, n_clusters, dim=0)
	ebr = torch.reshape(e, (batch_size, n_clusters, embedding_size))
	cbr = cluster_reps.repeat(batch_size,1,1)
	dist = torch.sum((ebr-cbr)**2,2)
	dist.data[dist<0] = 1

	a2 = torch.pow(dist,power_k)
	a2.data[a2 == float('inf')] = 1
	a2.data[a2 == float('-inf')] = -1
	a3 = torch.mean(a2,1)
	a3.data[a3<0] = 1
	Kolmo_mean = torch.pow(a3,1/power_k)
	Kolmo_mean.data[Kolmo_mean == float('inf')] = 1
	Kolmo_mean.data[Kolmo_mean == float('-inf')] = -1
	dist.register_hook(lambda x: torch.clamp(x, min=-100, max=100))
	'''
	cbr.register_hook(lambda x: print(math.isinf(float(torch.sum(x)))))
	ebr.register_hook(lambda x: print(math.isinf(float(torch.sum(x)))))
	dist.register_hook(lambda x: print(math.isinf(float(torch.sum(x)))))    
	e.register_hook(lambda x: print(math.isinf(float(torch.sum(x)))))   
	cluster_reps.register_hook(lambda x: print(math.isinf(float(torch.sum(x)))))
	embeddings.register_hook(lambda x: print(math.isinf(float(torch.sum(x)))))
	Kolmo_mean.register_hook(lambda x: print(math.isinf(float(torch.sum(x)))))
	'''

	'''
	cbr.register_hook(lambda x: torch.clamp(x, min=-100, max=100))
	ebr.register_hook(lambda x: torch.clamp(x, min=-100, max=100))
	dist.register_hook(lambda x: torch.clamp(x, min=-100, max=100))    
	e.register_hook(lambda x: torch.clamp(x, min=-100, max=100))   
	cluster_reps.register_hook(lambda x: torch.clamp(x, min=-100, max=100))
	embeddings.register_hook(lambda x: torch.clamp(x, min=-100, max=100))
	a2.register_hook(lambda x: torch.clamp(x, min=-100, max=100))
	a3.register_hook(lambda x: torch.clamp(x, min=-100, max=100))       
	Kolmo_mean.register_hook(lambda x: torch.clamp(x, min=-100, max=100))
	'''
	'''
	cbr.register_hook(lambda x: print("cbr 1",math.isinf(float(torch.sum(x)))))
	ebr.register_hook(lambda x: print("ebr 1",math.isinf(float(torch.sum(x)))))
	dist.register_hook(lambda x: print("dist 1",math.isinf(float(torch.sum(x)))))    
	e.register_hook(lambda x: print("e 1",math.isinf(float(torch.sum(x)))))   
	cluster_reps.register_hook(lambda x: print("cl 1",math.isinf(float(torch.sum(x)))))
	embeddings.register_hook(lambda x: print("emb 1",math.isinf(float(torch.sum(x)))))
	Kolmo_mean.register_hook(lambda x: print("km 1",math.isinf(float(torch.sum(x)))))
	a2.register_hook(lambda x: print("a2 1",math.isinf(float(torch.sum(x)))))
	a3.register_hook(lambda x: print("a3 1",math.isinf(float(torch.sum(x))))) 

	cbr.register_hook(lambda x: print("cbr 2",math.isnan(float(torch.sum(x)))))
	ebr.register_hook(lambda x: print("ebr 2",math.isnan(float(torch.sum(x)))))
	dist.register_hook(lambda x: print("dist 2",math.isnan(float(torch.sum(x)))))    
	e.register_hook(lambda x: print("e 2",math.isnan(float(torch.sum(x)))))   
	cluster_reps.register_hook(lambda x: print("cl 2",math.isnan(float(torch.sum(x)))))
	embeddings.register_hook(lambda x: print("emb 2",math.isnan(float(torch.sum(x)))))
	Kolmo_mean.register_hook(lambda x: print("km 2",math.isnan(float(torch.sum(x)))))
	a2.register_hook(lambda x: print("a2 2",math.isnan(float(torch.sum(x)))))
	a3.register_hook(lambda x: print("a3 2",math.isnan(float(torch.sum(x)))))

	dist.register_hook(lambda x: print(x))    

	'''
	'''
	print(e)
	print(ebr)
	print(cbr)
	print(dist)
	print(embeddings)  
	'''
	mean_batch_loss = torch.mean(Kolmo_mean)

	return mean_batch_loss
    