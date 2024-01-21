import wandb
import argparse

def wandb_init(
    config: argparse.Namespace,
    name: str
    ):
    wandb.init(
        project="robgen-2024-icml",
        config = vars(config),
        name=name
    )

def wandb_end():
    wandb.finish()
