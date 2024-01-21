from utils.utils import default_config, add_dict_to_argparser, create_model_from_config, load_dataset, loss_l1, CustomDataset
from utils.split_partitions import *
from torch.utils import data
from torchvision import transforms
from utils.wandb_utils import wandb_init, wandb_end

import torch
import argparse
import numpy as np
import wandb

num_clusters =  [100, 1000, 5000, 10000]

def cal_robustness(args):
    name = [str(value) for value in vars(args).values()]
    wandb_init(
        args,
        name="robustness_{}".format("_".join(name))
    )
    model = create_model_from_config(args)
    if args.model not in ["resnet18_imagenet", "regnet_imagenet"]:
        model_checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
        model.load_state_dict(model_checkpoint["state_dict"])

    train_dataset, val_dataset = load_dataset(args.dataset)
    if isinstance(train_dataset, CustomDataset):
        length_of_data = len(train_dataset.x)
        idx = np.random.choice(np.arange(length_of_data), 50000, replace=True)
        train_dataset = CustomDataset(
            x=train_dataset.x[idx],
            y=np.asarray(train_dataset.y)[idx],
            transform=transforms.ToTensor()
        )
    train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    val_dataloader = data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    loss_func = loss_l1

    loss = []
    train_loss = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # loss_func.to(device)
    model.eval()

    ## Feed dataset through nn
    with torch.no_grad():
        for batch in val_dataloader:
            output = model(batch[0].to(device))
            loss.append(torch.Tensor(loss_func(output, batch[1].to(device))).detach().cpu())
        for batch in train_dataloader:
            output = model(batch[0].to(device))
            train_loss.append(torch.Tensor(loss_func(output, batch[1].to(device))).detach().cpu())
    loss = torch.concat(loss)
    train_loss = torch.concat(train_loss)

    robustness = []
    sum_of_local_robustness = []

    for num_cluster in num_clusters:
        total_loss_subtraction = torch.abs(torch.cdist(loss.reshape(-1, 1), train_loss.reshape(-1, 1), p=1))
        a_robustness = torch.max(total_loss_subtraction.reshape(-1)).item()

        temp_sum_of_local_robustness = []

        for _ in range(10):
            centroids = select_partition_centroid(num_cluster, val_dataset)
            train_indices = assign_partition(train_dataset, centroids)
            test_indices = assign_partition(val_dataset, centroids)
            max_index = torch.max(test_indices)
            train_cluster_shape = []
            local_robustness = []
            
            for i in range(max_index + 1):
                train_loss_values = train_loss[(train_indices==i).nonzero()]
                loss_values = loss[(test_indices==i).nonzero()]
            
                if loss_values.shape[0] < 1 or train_loss_values.shape[0] < 1:
                    continue
                train_cluster_shape.append(train_loss_values.shape[0])
                loss_subtraction = torch.abs(torch.cdist(loss_values, train_loss_values, p=1))
                a_local_robustness = torch.max(loss_subtraction.reshape(-1)).item()
                local_robustness.append(a_local_robustness)

            a_sum_of_local_robustness = np.sum(np.array(train_cluster_shape) * np.array(local_robustness)) / np.sum(train_cluster_shape)
            temp_sum_of_local_robustness.append(a_sum_of_local_robustness)

        robustness.append(f"{a_robustness}")
        sum_of_local_robustness.append(f"{torch.mean(torch.Tensor(temp_sum_of_local_robustness)).item()}+-{torch.var(torch.Tensor(temp_sum_of_local_robustness)).item()}")
        wandb.log({
            "num_cluster": num_cluster,
            "robustness": a_robustness,
            "local_robustness": torch.mean(torch.Tensor(temp_sum_of_local_robustness)).item()
        })

    print(f"Robustness {robustness}")
    print(f"Summation of local robustness {sum_of_local_robustness}")
    wandb_end()
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())

    parser.add_argument("--model_checkpoint", type=str, default="model_best.pth.tar")
    args = parser.parse_args()
    cal_robustness(args)
