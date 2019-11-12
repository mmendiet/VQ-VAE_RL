import torch
import pickle
from collections import defaultdict
from torch.utils import data
import os

from my_classes import Dataset


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")

# Parameters
params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 1

dictDir = 'test_dict/'
all_partition = defaultdict(list)
all_labels = defaultdict(list)
# Datasets
for dictionary in os.listdir(dictDir):
    dfile = open(dictDir+dictionary, 'rb')
    d = pickle.load(dfile)
    dfile.close()
    if("partition" in dictionary):
        for key in d:
            all_partition[key] += d[key]
    elif("labels" in dictionary):
        for key in d:
            all_labels[key] = d[key]
    else:
        print("Error: Unexpected data dictionary")
#partition = # IDs
#labels = # Labels

# Generators
training_set = Dataset(all_partition['train'], all_labels)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(all_partition['validation'], all_labels)
validation_generator = data.DataLoader(validation_set, **params)

# Loop over epochs
for epoch in range(max_epochs):
    print(epoch)
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        print(local_batch.shape)
        print(local_labels.shape)