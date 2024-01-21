from train import start_train
from argparse import Namespace
from model._mlp import MLP
from model._vgg import VGG
from model._resnet import ResNet, BasicBlock
from torch import nn


def test_mlp():
    mlp_no_batch_norm = {
        "model": "mlp",
        "model_type": 0,
        "norm_type": "batch",
        "clamp_value": -1
    }

    args = Namespace(**mlp_no_batch_norm)

    config, training_config = start_train(args, True)
    assert config == {
        "in_features": 3 * 32 * 32,
        "cfg": [1024, 1024, 512, 512, 256, 256, 128, 64],
        "norm_layer": None,
        "num_classes": 10
    }
    assert type(training_config["model"]) == MLP
    assert config["norm_layer"] == None
    assert training_config["log_file_name"] == "mlp_wo_norm.txt"

    mlp_batch_norm = {
        "model": "mlp",
        "model_type": 2,
        "norm_type": "batch",
        "clamp_value": -1
    }
    args = Namespace(**mlp_batch_norm)
    config, training_config = start_train(args, True)
    assert type(training_config["model"]) == MLP
    assert config["norm_layer"] == nn.BatchNorm1d
    assert training_config["log_file_name"] == "mlp_batch.txt"

    mlp_group_norm = {
        "model": "mlp",
        "model_type": 2,
        "norm_type": "group",
        "clamp_value": -1
    }
    args = Namespace(**mlp_group_norm)
    config, training_config = start_train(args, True)
    assert type(training_config["model"]) == MLP
    assert config["norm_layer"] == nn.GroupNorm
    assert training_config["log_file_name"] == "mlp_group.txt"

    mlp_layer_norm = {
        "model": "mlp",
        "model_type": 2,
        "norm_type": "layer",
        "clamp_value": -1
    }
    args = Namespace(**mlp_layer_norm)
    config, training_config = start_train(args, True)
    assert type(training_config["model"]) == MLP
    assert config["norm_layer"] == nn.LayerNorm
    assert training_config["log_file_name"] == "mlp_layer.txt"

def test_resnet():
    resnet_no_batch_norm = {
        "model": "resnet",
        "model_type": 0,
        "norm_type": "batch",
        "clamp_value": -1
    }

    args = Namespace(**resnet_no_batch_norm)

    config, training_config = start_train(args, True)
    assert config == {
        "block": BasicBlock,
        "layers": [2, 2, 2, 2],
        "norm_layer": None,
        "num_classes": 10
    }
    assert type(training_config["model"]) == ResNet
    assert config["norm_layer"] == None
    assert training_config["log_file_name"] == "resnet_wo_norm.txt"

    resnet_batch_norm = {
        "model": "resnet",
        "model_type": 2,
        "norm_type": "batch",
        "clamp_value": -1
    }
    args = Namespace(**resnet_batch_norm)
    config, training_config = start_train(args, True)
    assert type(training_config["model"]) == ResNet
    assert config["norm_layer"] == nn.BatchNorm2d
    assert training_config["log_file_name"] == "resnet_batch.txt"

    resnet_group_norm = {
        "model": "resnet",
        "model_type": 2,
        "norm_type": "group",
        "clamp_value": -1
    }
    args = Namespace(**resnet_group_norm)
    config, training_config = start_train(args, True)
    assert type(training_config["model"]) == ResNet
    assert config["norm_layer"] == nn.GroupNorm
    assert training_config["log_file_name"] == "resnet_group.txt"

    resnet_layer_norm = {
        "model": "resnet",
        "model_type": 2,
        "norm_type": "layer",
        "clamp_value": -1
    }
    args = Namespace(**resnet_layer_norm)
    config, training_config = start_train(args, True)
    assert type(training_config["model"]) == ResNet
    assert config["norm_layer"] == nn.LayerNorm
    assert training_config["log_file_name"] == "resnet_layer.txt"

def test_vgg():
    vgg_no_batch_norm = {
        "model": "vgg",
        "model_type": 0,
        "norm_type": "batch",
        "clamp_value": -1
    }

    args = Namespace(**vgg_no_batch_norm)

    config, training_config = start_train(args, True)
    assert config == {
        "cfg": "D",
        "norm_layer": None,
        "init_weights": True,
        "num_classes": 10
    }
    assert type(training_config["model"]) == VGG
    assert config["norm_layer"] == None
    assert training_config["log_file_name"] == "vgg_wo_norm.txt"

    vgg_batch_norm = {
        "model": "vgg",
        "model_type": 2,
        "norm_type": "batch",
        "clamp_value": -1
    }
    args = Namespace(**vgg_batch_norm)
    config, training_config = start_train(args, True)
    assert type(training_config["model"]) == VGG
    assert config["norm_layer"] == nn.BatchNorm2d
    assert training_config["log_file_name"] == "vgg_batch.txt"

    vgg_group_norm = {
        "model": "vgg",
        "model_type": 2,
        "norm_type": "group",
        "clamp_value": -1
    }
    args = Namespace(**vgg_group_norm)
    config, training_config = start_train(args, True)
    assert type(training_config["model"]) == VGG
    assert config["norm_layer"] == nn.GroupNorm
    assert training_config["log_file_name"] == "vgg_group.txt"

    vgg_layer_norm = {
        "model": "vgg",
        "model_type": 2,
        "norm_type": "layer",
        "clamp_value": -1
    }
    args = Namespace(**vgg_layer_norm)
    config, training_config = start_train(args, True)
    assert type(training_config["model"]) == VGG
    assert config["norm_layer"] == nn.LayerNorm
    assert training_config["log_file_name"] == "vgg_layer.txt"

