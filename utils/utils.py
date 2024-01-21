from os.path import isfile
from typing import Dict, Tuple, List, Iterator
from contextlib import contextmanager
from torch import nn
from torchvision.datasets.utils import extract_archive
from torchvision.datasets.imagenet import load_meta_file
from torchvision.transforms.transforms import ToTensor
from utils.constant import MODEL_CONFIG, MODEL_MAP
from torchvision import transforms, datasets
from torchvision.models import resnet18, ResNet18_Weights, regnet_y_400mf, RegNet_Y_400MF_Weights, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from PIL import Image

import argparse
import shutil
import tempfile
import torch
import numpy as np
import os
import pickle

META_FILE = "meta.bin"

class ImageNetDataset(datasets.ImageFolder):
    def __init__(
            self,
            root: str,
            split: str = 'train',
            **kwargs
        ):
        # root = self.root = os.path.expanduser(root)
        self.root = root
        self.test = root
        self.split = split,
        parse_dev_kit(root, "test")
        wnid_to_classes = load_meta_file("data")[0]
        wnids = load_meta_file("data")[1]

        self.split = self.split[0]
        if split == "val":
            images = sorted(os.path.join(self.split_folder, image) for image in os.listdir(self.split_folder))
            if len(images) >= 50000:
                for wnid in set(wnids):
                    os.mkdir(os.path.join(self.split_folder, wnid))

                for wnid, img_file in zip(wnids, images):
                    shutil.move(img_file, os.path.join(self.split_folder, wnid, os.path.basename(img_file)))
        super().__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}
        
    @property
    def split_folder(self)->str:
        return os.path.join(self.test, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

def parse_dev_kit(root: str, file: str):

    import scipy.io as sio

    def parse_meta_mat(devkit_root: str) -> Tuple[Dict[int, str], Dict[str, Tuple[str, ...]]]:
        metafile = os.path.join(devkit_root, "data", "meta.mat")
        meta = sio.loadmat(metafile, squeeze_me=True)["synsets"]
        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(", ")) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
        return idx_to_wnid, wnid_to_classes

    def parse_val_groundtruth_txt(devkit_root: str) -> List[int]:
        file = os.path.join(devkit_root, "data", "ILSVRC2012_validation_ground_truth.txt")
        with open(file) as txtfh:
            val_idcs = txtfh.readlines()
        return [int(val_idx) for val_idx in val_idcs]

    @contextmanager
    def get_tmp_dir() -> Iterator[str]:
        tmp_dir = tempfile.mkdtemp()
        try:
            yield tmp_dir
        finally:
            shutil.rmtree(tmp_dir)

    if file is None:
        raise ValueError()

    with get_tmp_dir() as tmp_dir:
        # extract_archive(os.path.join(root, file), tmp_dir)

        devkit_root = os.path.join('data', "ILSVRC2012_devkit_t12")
        idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
        val_idcs = parse_val_groundtruth_txt(devkit_root)
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

        torch.save((wnid_to_classes, val_wnids), os.path.join("data", META_FILE))

def default_config():
    return {
        "model": "mlp",
        "dataset": "CIFAR10",
        "model_type": 2,
        "clamp_value": -1.0,
        "norm_type": "BN",
        "from_checkpoint": False,
        "num_epoch": 20,
        "learning_rate": 0.01,
        "weight_decay": 1e-4,
        "optimizer": "sgd"
    }

def add_dict_to_argparser(
    parser: argparse.ArgumentParser,
    config_dict: Dict
    ):
    for k, v in config_dict.items():
        v_type = type(v)
        parser.add_argument(f"--{k}", type=v_type, default=v)

def create_model_from_config(
    args: argparse.Namespace
    ):
    if args.model == "resnet18_imagenet":
        model = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        return model
    elif args.model == "regnet_imagenet":
        model = regnet_y_400mf(RegNet_Y_400MF_Weights.IMAGENET1K_V1)
        return model
    elif args.model == "shufflenet1":
        return shufflenet_v2_x1_0()
    elif args.model == "shufflenet2":
        return shufflenet_v2_x1_5()
    elif args.model == "shufflenet3":
        return shufflenet_v2_x2_0()
    config = MODEL_CONFIG[args.model]
    if args.model_type == 2:
        if args.norm_type == "BN":
            if args.model == "mlp":
                config["norm_layer"] = nn.BatchNorm1d
            if args.model == "mlp_1d":
                config["norm_layer"] = nn.BatchNorm1d
            else:
                config["norm_layer"] = nn.BatchNorm2d
        
        elif args.norm_type == "GN":
            config["norm_layer"] = nn.GroupNorm
        elif args.norm_type == "LN":
            config["norm_layer"] = nn.LayerNorm
        else:
            raise NotImplementedError("This norm type has not been implemented yet.")
    elif args.model_type == 1 and args.model in ["resnet", "resnet34", "resnet50"]:
        config["signal"] = 1
    elif args.model_type == 1:
        raise NotImplementedError("This setting is not support for vgg and mlp")
    
    return MODEL_MAP[args.model](**config)

def load_dataset(
    dataset: str = "CIFAR10"
    ):
    if dataset == "IMAGENET_RESNET":
        train_dataset = ImageNetDataset(
            root="/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/",
            split="train",
            # transform=transforms.Compose([transforms.ToTensor(), transforms.Resize([64, 64])]),
            transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
        )
        val_dataset= ImageNetDataset(
            root="data/",
            split="val",
            # transform=transforms.Compose([transforms.ToTensor(), transforms.Resize([64, 64])])
            transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
        )
    elif dataset == "IMAGENET_REGNET":
        train_dataset = ImageNetDataset(
            root="/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/",
            split="train",
            # transform=transforms.Compose([transforms.ToTensor(), transforms.Resize([64, 64])]),
            transform=RegNet_Y_400MF_Weights.IMAGENET1K_V1.transforms()
        )
        val_dataset= ImageNetDataset(
            root="data/",
            split="val",
            # transform=transforms.Compose([transforms.ToTensor(), transforms.Resize([64, 64])])
            transform=RegNet_Y_400MF_Weights.IMAGENET1K_V1.transforms()
        )

    elif dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(root="cifar_train", 
                                         train=True, 
                                        transform=transforms.Compose([
                                        transforms.ToTensor()                                        ]),
                                        download=True)
        val_dataset = datasets.CIFAR10(root="cifar_val",
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    elif dataset == "CIFAR10_AUG10":
        if os.path.isfile("cifar10_aug10.pth"):
            # train_data = torch.load('cifar10_aug10.pth')
            with open("cifar10_aug10.pth", "rb") as f:
                train_data = pickle.load(f)
            train_dataset = CustomDataset(
                x=train_data["x"],
                y=train_data["y"],
                transform=transforms.ToTensor()
            )
        else:
            train_dataset = create_augmented_dataset("CIFAR10", 10)
            # torch.save({
            #     "x": train_dataset.x.clone(),
            #     "y": train_dataset.y.clone(),
            # }, "cifar10_aug10.pth")
            with open("cifar10_aug10.pth", "wb+") as f:
                pickle.dump({
                    "x": train_dataset.x,
                    "y": train_dataset.y
                }, f)
        val_dataset = datasets.CIFAR10(root="cifar_val",
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    elif dataset == "CIFAR10_AUG50":
        if os.path.isfile("cifar10_aug50.pth"):
            # train_data = torch.load('cifar10_aug50.pth')
            with open("cifar10_aug50.pth", "rb") as f:
                train_data = pickle.load(f)
            train_dataset = CustomDataset(
                x=train_data["x"],
                y=train_data["y"],
                transform=transforms.ToTensor()
            )
        else:
            train_dataset = create_augmented_dataset("CIFAR10", 50)
            print("train_dataset ok, start saving to file")
            # torch.save({
            #     "x": train_dataset.x,
            #     "y": train_dataset.y
            # }, "cifar10_aug50.pth")
            with open("cifar10_aug50.pth", "wb+") as f:
                pickle.dump({
                    "x": train_dataset.x,
                    "y": train_dataset.y
                }, f)
            print("end saving to file")
        val_dataset = datasets.CIFAR10(root="cifar_val",
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    elif dataset == "MNIST":
        train_dataset = datasets.MNIST(root="mnist_train", train=True, 
                                        transform=transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        ]),
                                        download=True)
        val_dataset = datasets.MNIST(root="mnist_val",
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    elif dataset == "SVHN":
        train_dataset = datasets.SVHN(root="svhn_train", split = 'train', 
                                        transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        ]),
                                        download=True)
        val_dataset = datasets.SVHN(root="svhn_val",
                                    split = "test",
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    elif dataset == "SVHN_AUG10":
        if os.path.isfile("svhn_aug10.pth"):
            # train_data = torch.load("SVHN_AUG10.pth")
            with open("svhn_aug10.pth", "rb") as f:
                train_data = pickle.load(f)
            train_dataset = CustomDataset(
                x=train_data["x"],
                y=train_data["y"],
                transform=transforms.ToTensor()
            )
        else:
            train_dataset = create_augmented_dataset("SVHN", 10)
            # torch.save({
            #     "x": train_dataset.x,
            #     "y": train_dataset.y
            # }, "svhn_aug10.pth")
            with open("svhn_aug10.pth", "wb+") as f:
                pickle.dump({
                    "x": train_dataset.x,
                    "y": train_dataset.y
                }, f)
        val_dataset = datasets.SVHN(root="svhn_val",
                                    split = "test",
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    elif dataset == "SVHN_AUG50":
        if os.path.isfile("svhn_aug50.pth"):
            # train_data = torch.load("SVHN_AUG50.pth")
            with open("svhn_aug50.pth", "rb") as f:
                train_data = pickle.load(f)
            train_dataset =  CustomDataset(
                x=train_data["x"],
                y=train_data["y"],
                transform=transforms.ToTensor()
            )
        else:
            train_dataset = create_augmented_dataset("SVHN", 50)
            # torch.save({
            #     "x": train_dataset.x,
            #     "y": train_dataset.y
            # }, "svhn_aug50.pth")
            with open("svhn_aug50.pth", "wb+") as f:
                train_data = pickle.dump({
                    "x": train_dataset.x,
                    "y": train_dataset.y
                }, f)
        val_dataset = datasets.SVHN(root="svhn_val",
                                    split = "test",
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)

    elif dataset == "Fashion_MNIST":
        train_dataset = datasets.FashionMNIST(root="fashion_mnist_train", train=True, 
                                        transform=transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        ]),
                                        download=True)
        val_dataset = datasets.FashionMNIST(root="fashion_mnist_val",
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    return train_dataset, val_dataset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x,
        y,
        transform,
        target_transform=None,
        **kwargs
    ):
        self.x = x
        self.y = y
        self.transform = transform
        self.target_transform = target_transform
        super().__init__(**kwargs)

    def __getitem__(self, index):
        img, label = self.x[index], self.y[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.x)

def create_augmented_dataset(
    original_dataset: str = "CIFAR10",
    num_augmented: int = 10
    ):
    if original_dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(root="cifar_train", 
                                         train=True, 
                                        transform=transforms.Compose([
                                        transforms.ToTensor()                                        ]),
                                        download=True)
        label = [train_dataset.targets]
    elif original_dataset == "SVHN":
        train_dataset = datasets.SVHN(root="svhn_train", split = 'train', 
                                        transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        ]),
                                        download=True)
        label = [train_dataset.labels]
    augmented_data = [train_dataset.data.reshape(-1, 3, 32, 32)]
    print("Start augmenting data")
    for _ in range(num_augmented):
        augmented_data.append(transforms.RandAugment()(torch.tensor(augmented_data[0], dtype=torch.uint8)).numpy())
        label.append(label[0])
    print("Ending augmenting data")
    label = np.concatenate(np.array(label)).tolist()
    print("Ending concatenating label")
    augmented_data = np.concatenate(augmented_data).reshape(-1, 32, 32, 3)
    print("Ending concatenating data")
    return CustomDataset(
        x = augmented_data,
        y = label,
        transform = transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform = None
    )

def loss_l1(y_pred, y_true):
    label_pred = torch.argmax(y_pred, dim=1)
    loss = (label_pred != y_true).int().float() * 2
    return loss

def loss_l2(weight, batch):
    batch = batch.reshape(batch.shape[0], -1)
    weight = weight.T
    return 1.0 / np.sum(np.power(np.dot(batch, weight), 2), axis=1)
