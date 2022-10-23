# Framework-for-Deep-Power-K-means

## Introduction

This is an experimental framework to jointly learn cluster representations and optimise the power K means objective. This work is inspired by the two papers :  
* [Power k-Means Clustering](http://proceedings.mlr.press/v97/xu19a/xu19a.pdf). 
----
![powerk](extras/powerk.png). 

Power K-means tries to optimise the k-means objective iteratively through a series of smoother power-mean objectives, using the Majorisation-Minimisation principle.  


* [Deep k-Means: Jointly clustering with k-Means and learning representations](https://arxiv.org/pdf/1806.10069.pdf)
----
![deepk](extras/deepk.png). 
In deep K-means, low dimensional cluster representations are jointly learned using an Auto-encoder loss and a differentiable surrogate for the K-means objective.  


In this implementation, I have tried to jointly optimise the power means objective together with the auto-encoder loss for each iteration of the power k-means algorithm. Following Deep K-means, the power-mean loss in each iteration is computed on the low dimensional cluster and data representations.

While tuning parameters, please keep in mind that the starting value of the Power_k param should be < -1 to ensure concavity of the loss surface and facilitate the learning process.(Proof can be found in the Power K-means paper.)



## Dependencies
```
* Python
* Pytorch
* numpy
* scipy
* scikit-learn
* pickle
* matplotlib
```

## Run on your dataset

1. Store your dataset in a .npz file with :  
  * _dataset['feature'] (num_samples, feat_dim)_
  * _dataset['target'] (num_samples,)_
  
2. Run the run.py with input_path for the dataset.npz file and output path for the results and plots. Other hyper-parameters as modified as required. The complete list of tunable parameters can be found in opts.py.  
```
python run.py --input_path --output_path --plot_path
```
