#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:26:47 2020

@author: kratochvila
"""
import torch
import numpy as np

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD import DeepSVDD
from datasets.main import load_dataset

net_name="cifar10_LeNet"
load_model="/model.tar"
dataset_name="mydata"
xp_path="../log/mydata_test"

cfg = Config(locals().copy())
deep_SVDD = DeepSVDD('one-class', 0.1)
deep_SVDD.set_network(net_name)
# If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
deep_SVDD.load_model(model_path=xp_path+load_model, load_ae=False)

dataset = load_dataset(dataset_name, '../data', 1)
deep_SVDD.test(dataset, device='cpu', n_jobs_dataloader=0)

# Plot most anomalous and most normal (within-class) test samples
indices, labels, scores = zip(*deep_SVDD.results['test_scores'])
indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score
idx_sorted_wrong = indices[labels == 1][np.argsort(scores[labels == 1])]
if dataset_name in ('mnist', 'cifar10', 'mydata'):

    if dataset_name == 'mnist':
        X_normals = dataset.test_set.test_data[idx_sorted[:32], ...].unsqueeze(1)
        X_outliers = dataset.test_set.test_data[idx_sorted[-32:], ...].unsqueeze(1)

    if dataset_name == 'cifar10':
        X_normals = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[:32], ...], (0, 3, 1, 2)))
        X_outliers = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[-32:], ...], (0, 3, 1, 2)))
    
    if dataset_name == 'mydata':
        X_normals = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[:32], ...], (0, 3, 1, 2)))
        X_outliers = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[-32:], ...], (0, 3, 1, 2)))
        X_wrong_normals = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted_wrong[:32], ...], (0, 3, 1, 2)))
        X_wrong_outliers = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted_wrong[-32:], ...], (0, 3, 1, 2)))

    plot_images_grid(X_normals,title='Most normal examples', padding=2, nrow=4) # export_img=xp_path + '/normals', 
    plot_images_grid(X_outliers, title='Most anomalous examples', padding=2, nrow=4) # export_img=xp_path + '/outliers',
    if dataset_name == 'mydata':
        plot_images_grid(X_wrong_normals, title='Most normal examples', padding=2, nrow=4) # export_img=xp_path + '/wrong_normals',
        plot_images_grid(X_wrong_outliers, title='Most anomalous examples', padding=2, nrow=4) # export_img=xp_path + '/wrong_outliers',

print("Test AUC: {0:.3f}".format(deep_SVDD.results['test_auc']))
# plot_images_grid(torch.tensor(np.transpose(dataset.test_set.test_data[0],(2,0,1))),nrow=1)