import torch
import numpy as np
from torch.utils import data

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = np.load('test_data/' + ID + '.npz')
        X = torch.from_numpy(X['arr_0']).float().permute(2,0,1)
        X[0:12,:,:] = X[0:12,:,:]/256
        #shape = X[12,:,:]
        #X[12,:,:] = X[12,:,:]
        outFile = self.labels[ID]
        y = np.load('test_data/' + outFile + '.npz')
        y = torch.from_numpy(y['arr_0']).float().permute(2,0,1)/256
        #y[0:3,:,:] = y[0:3,:,:]/256
        #shape2 = y[3,:,:].shape
        #y[3,:,:] = y[3,:,:]/(shape2[0]*shape2[1])

        return X, y