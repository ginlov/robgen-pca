from utils.train_base import train
from utils.utils import default_config, add_dict_to_argparser, create_model_from_config, load_dataset, loss_l2
from utils.split_partitions import select_partition_centroid, assign_partition
from utils.wandb_utils import wandb_init, wandb_end
from torch.utils import data
from sklearn.decomposition import PCA

import argparse
import numpy as np
import torch

num_cluster = 10000
sigma = [0.01, 0.05, 0.1]
num_component = 300

def start_train(
        args: argparse.Namespace,
    ):
    # create model
    model = PCA(n_components=num_component)
    train_dataset, valid_dataset = load_dataset(args.dataset)
    num_items = len(train_dataset)

    dataloader = data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    for batch in dataloader:
        input_features = batch[0].detach().numpy()
    input_features = input_features.reshape(input_features.shape[0], -1)
    model.fit(input_features)
    return model


def cal_g3(k, sigma, total_num_items, cluster_num_item, list_of_a, TS):
    a0 = max(list_of_a)
    k = torch.tensor(k)
    sigma = torch.tensor(sigma)
    total_num_items = torch.tensor(total_num_items)
    cluster_num_item = torch.tensor(cluster_num_item)
    list_of_a = torch.tensor(list_of_a)
    TS = torch.tensor(TS)
    g3_first = torch.sqrt(torch.log(2*k/sigma)) / total_num_items * torch.sum(torch.sqrt(cluster_num_item)*(a0 + torch.sqrt(torch.tensor(2))*list_of_a))
    g3_second = 2*torch.log(2*k / sigma) / total_num_items * (a0 * TS + torch.sum(list_of_a))
    return g3_first + g3_second


def the_rest_of_theorem_five(list_of_a, list_of_local_loss, list_of_num_item, num_items):
    return torch.sum(torch.tensor(list_of_num_item) * (torch.tensor(list_of_a) - torch.tensor(list_of_local_loss))) / num_items


def cal_g(model, args):
    train_dataset, valid_dataset = load_dataset(args.dataset)
    num_items = len(train_dataset)
    train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=False)

    valid_dataloader = data.DataLoader(valid_dataset, batch_size=128, shuffle=False)

    ## Set up loss function, device, ...
    loss_func = loss_l2 ## Replace with l2 loss function
    train_loss = []
    valid_loss = []

    for batch in train_dataloader:
        train_loss.append(loss_func(model.components_, batch[0].detach().numpy()))

    for batch in valid_dataloader:
        valid_loss.append(loss_func(model.components_, batch[0].detach().numpy()))

    train_loss = np.concatenate(train_loss)
    valid_loss = np.concatenate(valid_loss)

    C_temp = max(valid_loss)
    C = max(train_loss)
    C = max(C_temp, C)

    print("Average train loss by L2 loss: {}".format(np.mean(train_loss)))

    list_of_num_item = []
    list_of_a = []
    list_of_local_loss = []

    local_robustness_cluster_shape = []
    local_robustness = []
    
    centroids = select_partition_centroid(num_cluster, valid_dataset)
    train_indices = assign_partition(train_dataset, centroids)
    valid_indices = assign_partition(valid_dataset, centroids)

    unique_ids = torch.unique(train_indices)
    for each in unique_ids:
        cluster_loss = train_loss[train_indices == each]
        cluster_valid_loss = valid_loss[valid_indices == each]
        list_of_num_item.append(cluster_loss.shape[0])
        list_of_local_loss.append(cluster_loss.mean().item())
        if cluster_valid_loss.shape[0] != 0:
            list_of_a.append(np.concatenate([cluster_loss, cluster_valid_loss], axis=0).mean())

            #local robustness
            local_robustness_cluster_shape.append(cluster_loss.shape[0])
            loss_subtraction = torch.abs(torch.cdist(torch.tensor(cluster_valid_loss).reshape(-1, 1), torch.tensor(cluster_loss).reshape(-1, 1)))
            a_local_robustness = torch.max(loss_subtraction.reshape(-1)).item()
            local_robustness.append(a_local_robustness)
        else:
            list_of_a.append(cluster_loss.mean())

    TD = unique_ids.shape[0]
    g_value = []
    for sigma_value in sigma:
        g_value.append(cal_g3(k=num_cluster, sigma=sigma_value, total_num_items=num_items, cluster_num_item=list_of_num_item, list_of_a=list_of_a, TS=TD))

    total_loss_subtraction = torch.abs(torch.cdist(torch.tensor(valid_loss).reshape(-1, 1), torch.tensor(train_loss).reshape(-1, 1)))
    robustness = torch.max(total_loss_subtraction.reshape(-1)).item()
    local_robustness = np.sum(np.array(local_robustness_cluster_shape)*np.array(local_robustness)) / np.sum(local_robustness_cluster_shape)

    the_rest_theorem_five = the_rest_of_theorem_five(list_of_local_loss=list_of_local_loss,
                                                list_of_a = list_of_a,
                                                list_of_num_item=list_of_num_item,
                                                num_items=num_items)

    print(f"""
    g value {sigma[0]} {g_value[0]},
    g value {sigma[1]} {g_value[1]},
    g value {sigma[2]} {g_value[2]},
    robustness {robustness},
    local_robustness {local_robustness},
    the_rest_theorem_five {the_rest_theorem_five}
    """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())
    
    args = parser.parse_args()

    model = start_train(args)

    cal_g(model, args)

