from utils.train_base import train
from utils.utils import default_config, add_dict_to_argparser, create_model_from_config
from utils.wandb_utils import wandb_init, wandb_end

import argparse

def start_train(
        args: argparse.Namespace,
    ):
    # create model
    model = create_model_from_config(args)
    dataset = args.dataset
    name = [str(value) for value in vars(args).values()]
    wandb_init(
        config=args,
        name="train_{}".format("_".join(name))
    )

    # training
    if args.model_type == 0:
        norm_type = "wo_norm"
    else:
        norm_type = args.norm_type
    log_file_name = "_".join([args.model, norm_type]) + ".txt"
    log_folder = "_".join([args.model, norm_type])
    training_config = {
        "model": model,
        "dataset": dataset,
        "log_file_name": log_file_name, 
        "clamp_value": args.clamp_value,
        "from_checkpoint": args.from_checkpoint,
        "log_folder": log_folder,
        "epoch": args.num_epoch,
        "learning_rate": args.learning_rate,
        "config_weight_decay": args.weight_decay,
        "config_optimizer": args.optimizer
    }

    train(**training_config)

    wandb_end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())
    
    args = parser.parse_args()

    start_train(args)
