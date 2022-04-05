import random
import numpy as np
import pickle
import time
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import os
import math 
from sklearn.datasets import fetch_openml 
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from model import PowerKdeep
from loss import AE_loss, PowerK_loss, AE_loss_mod, PowerK_loss_mod, cluster_acc
from dataset import deepKp
import opts


def run_deepKP(data, target, opt, power_k):



	os.makedirs(opt['output_path'], exist_ok = True)

	pkd = PowerKdeep(500,500,2000,opt['embedding_size'],opt['n_clusters']).to(device)
	optm = optim.Adam(pkd.parameters(), lr=opt['learning_rate'])

	mnist_data = deepKp(data, target)
	mnsitloader = DataLoader(mnist_data, opt['batch_size'], shuffle=False, drop_last=True, num_workers=2)


	flag = False
	clusterassgnperS = []
	graddict = {}

	for n,p in pkd.named_parameters():
	    graddict[n]=[]
	torch.backends.cudnn.enabled = False

	AEl = []
	PKl = []
	Es = []
	decr = []
	encr = []
	cluster_rep_per_epoch = []


	for iterations in range(opt['maxSvalues']):

		for epoch in range(opt['max_epochs']):

			running_loss =0
		    running_lossae =0
		    running_losspk =0
		    running_emb =0

		    cluster_assignments = []
		    embdata = []
		    gt = []

		    for i_batch, (image, label) in enumerate(mnsitloader):

		    	#print(image.shape)
				#print(label.shape)
				#print(i_batch)



				image = image.to(device)
				image = image.float()
				#with torch.autograd.detect_anomaly():
				embedding,cluster_rep,reconstr_out = pkd(image)
				#embedding.register_hook(lambda x: torch.clamp(x, min=-100, max=100))
				#cluster_rep.register_hook(lambda x: torch.clamp(x, min=-100, max=100))

				if (i_batch > 0) :
					if (torch.all(torch.eq(o, cluster_rep))):
				  		print("not updated")
				o = cluster_rep.clone().detach()#torch.clone(cluster_rep)

				#print(cluster_rep)
				aeloss = AE_loss(reconstr_out,image)
				pkloss = PowerK_loss(embedding,cluster_rep,power_k)
				regloss = None
				regloss1 = None

				for n,p in pkd.named_parameters():
			        if "dec" in n :
						if regloss is None:
							regloss = torch.sum(p**2)
						else:
							regloss = regloss + torch.sum(p**2)

				for n,p in pkd.named_parameters():
					if "enc" in n :
				    	if regloss1 is None:
					        	regloss1 = torch.sum(p**2)
				    	else:
					    	regloss1 = regloss1 + torch.sum(p**2)

				l_pk = opt['lambda_pk']
				l_reg = opt['lambda_reg']
				loss = aeloss + l_pk*pkloss + l_reg*regloss
				loss.backward()
				#print(loss)
				#print(aeloss)
				#print(pkloss)

				#torch.nn.utils.clip_grad_value_(pkd.parameters(), 100)


				for n,p in pkd.named_parameters():
			        if (math.isnan(float(torch.sum(p.grad)))):
						print(n," has nan grad in batch",i_batch+1)
						print(p.grad)
						flag = True

						'''  
						elif (n==("cluster_reps")):
						print("c:\n",p.grad)
						'''
			        else:
			        	graddict[n].append(p.grad)

			    if (flag):
        			break
        
      			optm.step()
      			optm.zero_grad()

      			#print("S :",power_k,"E :",epoch + 1,"B :", i_batch +1, "loss :",float(loss), "aeloss :",float(aeloss) ,"pkloss :",float(pkloss))
				running_loss += float(loss)
				running_lossae += float(aeloss)
				running_losspk += float(pkloss)

				emb = embedding.clone().detach()
				cent = cluster_rep.clone().detach()
				with torch.no_grad():
					e = torch.repeat_interleave(emb, opt['n_clusters'], dim=0)
					ebr = torch.reshape(e, (opt['batch_size'], opt['n_clusters'], opt['embedding_size']))
					cbr = cent.repeat(opt['batch_size'],1,1)
					dist = torch.sum((ebr-cbr)**2,2)
					asgn = torch.argmin(dist, dim=1)
					#l = np.asarray(label)
					#print(label.dtype)
					gt.append(label)
					cluster_assignments.append(asgn)
					embdata.append(emb)
					#running_emb += float(emb)

			if (flag):
      			break

      		embD = torch.cat(embdata, dim = 0)
		    #print(embD.shape)
		    with torch.no_grad():
				emean = torch.mean(embD,0)
				#print(emean.shape)
				emr = emean.repeat(70000,1)
				#print(emr.shape)
				subspacesize = torch.sum((emr-embD)**2)
				#print(subspacesize)


			print("S :",power_k,"epoch ",epoch + 1, "loss :",running_loss, "aeloss :",running_lossae, "pkloss :", running_losspk,"size :",float(subspacesize),"regloss :",float(regloss),"regloss1 :",float(regloss1))
		    print(cent)
		    cluster_rep_per_epoch.append(cent)
		    AEl.append(running_lossae)
		    PKl.append(running_losspk)
		    Es.append(float(subspacesize))
		    decr.append(float(regloss))
		    encr.append(float(regloss1))

		    #print("loss :",running_loss)
		    #print("aeloss :",running_lossae)
		    #print("pkloss :",running_losspk)
		    pred = torch.cat(cluster_assignments, dim = 0)
		    labels = torch.cat(gt, dim = 0)
		    
		    #print(pred.shape)

		if (flag):
    		break

    	print_val = 1
		if iterations % print_val == 0 or iterations == opt['maxSvalues'] - 1:
			ytrue = np.asarray(labels)
			print(target)
			print(labels)
			ypred = np.asarray(pred)
			clusterassgnperS.append(ypred)
			print("results after iteration ", iterations +1)
			print("acc",cluster_acc(ytrue,ypred))
			ari = adjusted_rand_score(ytrue,ypred)
			print("ARI", ari)
			nmi = normalized_mutual_info_score(ytrue,ypred)
			print("NMI", nmi)
			state = {'powerk_value': power_k, 'state_dict': pkd.state_dict()}
			torch.save(state, output_path + f"/pkd_checkpoint_{iterations}.pth.tar")

		power_k = (opt['p_rate'])*power_k


	output_path = opt['output_path']

	with open(output_path + 'predperS.txt', 'wb') as fp:
		pickle.dump(clusterassgnperS,fp)
	with open(output_path + 'crepperE.txt', 'wb') as fp:
		pickle.dump(cluster_rep_per_epoch,fp)
	with open(output_path + 'graddict.pickle', 'wb') as handle:
		pickle.dump(graddict, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open(output_path + 'aeloss.txt', 'wb') as fp:
		pickle.dump(AEl,fp)
	with open(output_path + 'pkloss.txt', 'wb') as fp:
		pickle.dump(PKl,fp)
	with open(output_path + 'Embsize.txt', 'wb') as fp:
		pickle.dump(Es,fp)
	with open(output_path + 'decreg.txt', 'wb') as fp:
		pickle.dump(decr,fp)
	with open(output_path + 'encreg.txt', 'wb') as fp:
		pickle.dump(encr,fp)




def plot(opt):

	plot_path = opt['plot_path']
	output_path = opt['output_path']
	os.makedirs(opt['plot_path'], exist_ok = True)



	plt.rcParams["figure.figsize"] = (10,5)

	with open(output_path + 'pkloss.txt', 'rb') as fp:
		l1 = pickle.load(fp)
	color = 'black'
	with plt.rc_context({'xtick.color': color, 'ytick.color': color}):
		a, = plt.plot(l1, label = 'Powerkloss')
		plt.legend(handles=[a])
		plt.savefig(plot_path + 'Powerkloss.png')
		plt.clf()

	with open(output_path + 'Embsize.txt', 'rb') as fp:
		l2 = pickle.load(fp)
	color = 'black'
	with plt.rc_context({'xtick.color': color, 'ytick.color': color}):
		b, = plt.plot(l2, label = 'Embsize')
		plt.legend(handles=[b])
		plt.savefig(plot_path + 'Embsize.png')
		plt.clf()

	with open(output_path + 'decreg.txt', 'rb') as fp:
		l1 = pickle.load(fp)
	with open(output_path + 'encreg.txt', 'rb') as fp:
		l2 = pickle.load(fp)
	color = 'black'
	with plt.rc_context({'xtick.color': color, 'ytick.color': color}):
		a, = plt.plot(l1, label = 'Decoder')
		b, = plt.plot(l2, label = 'Encoder')
		plt.legend(handles=[a, b])
		plt.savefig(plot_path + 'reg_loss.png')
		plt.clf()

	with open(output_path + 'aeloss.txt', 'rb') as fp:
		l2 = pickle.load(fp)
	color = 'black'
	with plt.rc_context({'xtick.color': color, 'ytick.color': color}):
		a, = plt.plot(l2, label = 'Aeloss')
		plt.legend(handles=[a])
		plt.savefig(plot_path + 'AutoEncloss.png')
		plt.clf()




if __name__ == '__main__':


	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	opt = opts.parse_opt()
	opt = vars(opt)


	# read features and labels
	input_path = opt['input_path']
	dataset = np.load(input_path) #.npz file containing feature((num_features, feature dim) and target((num_features,))
	data = dataset['feature']
	target = dataset['target']

	run_deepKP(data, target, opt, opt['power_k'])
	plot(opt)



	









