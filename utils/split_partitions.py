from torch.utils import data
from torch import Tensor
from typing import List
from tqdm import tqdm

import torch
import random
import numpy as np

class PseudoDataset(data.Dataset):
    def __init__(
                self,
                X,
                y
        ):
        super(PseudoDataset).__init__()
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

def select_partition_centroid(
    num_partitions: int,
    train_dataset: data.Dataset
    ):
    if len(train_dataset) == num_partitions:
        index_choices = list(range(num_partitions))
    index_choices = random.sample(range(len(train_dataset)), 
                                   k=num_partitions)
    centroids = []
    for each in index_choices:
        centroids.append(train_dataset[each][0].reshape(-1))
    return torch.vstack(centroids)

def assign_partition(
    test_dataset: data.Dataset,
    centroids: Tensor
    ):
    data_loader = data.DataLoader(
        test_dataset, 
        batch_size=512,
        shuffle=False)
    list_of_indices = []
    num_of_batch = len(data_loader)
    for id_, batch in tqdm(enumerate(data_loader)):
        features_data = batch[0]
        batch_size, _, __, ___ = features_data.shape
        distance = torch.cdist(features_data.reshape(batch_size, -1), centroids, p=2)
        cluster_indices = torch.argmin(distance, dim=1)
        list_of_indices.append(cluster_indices)
    return torch.concat(list_of_indices)

def calculate_robustness(
    model: torch.nn.Module,
    train_dataset: data.Dataset,
    test_dataset: data.Dataset,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
    loss_func,
    ):
    train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    dataloader = data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    loss = []
    train_loss = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch[0])
            loss.append(torch.Tensor(loss_func(output, batch[1])))
        for batch in train_dataloader:
            output = model(batch[0])
            train_loss.append(torch.Tensor(loss_func(output, batch[1])))
    loss = torch.concatenate(loss)
    train_loss = torch.concatenate(train_loss)
    max_index = torch.max(test_indices)
    epsilon = 0.0
    for i in range(max_index + 1):
        train_loss_values = train_loss[(train_indices==i).nonzero()]
        loss_values = loss[(test_indices==i).nonzero()]
        
        if loss_values.shape[0] < 1 or train_loss_values.shape[0] < 1:
            continue
        loss_subtraction = torch.abs(torch.cdist(loss_values, train_loss_values, p=1))
        epsilon = max(epsilon, torch.max(loss_subtraction.reshape(-1)).item())

    return epsilon

def generate_dataset(
    num_sample: int = 10000,
    num_features: List[int] = [3, 32, 32],
    ):
    X = np.random.uniform(0.0, 1.0, [num_sample] + num_features)
    y = np.random.uniform(0.0, 1.0, [num_sample])
    return PseudoDataset(X, y)
