from model._resnet import ModifiedLinear, ModifiedConv2d, ModifiedMaxPool2d, ModifiedAdaptiveAvgPool2d
from torchvision import datasets
from enum import Enum
from torch import nn
from loguru import logger
from torch.utils import data

import torch
import time
import shutil
import numpy as np
import os
import wandb

from utils.utils import load_dataset


def train(model, dataset, log_file_name="", log_folder="log", clamp_value=-1, from_checkpoint=False, epoch=20, learning_rate=0.01, config_weight_decay=1e-4, config_optimizer="sgd"):
    ##############################
    ###### Settings ##############
    ##############################
    global best_acc1
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_loss = 0.0
    train_best_acc1 = 0.0
    train_best_acc5 = 0.0
    train_best_loss = 0.0
    patient = 0
    batch_size = 64
    workers = 5
    lr = learning_rate
    # num_epoch = 1
    num_epoch = epoch 
    weight_decay = config_weight_decay
    momentum = 0.9
    max_patient = 3

    ###############################
    ### LOADING DATASET ###########
    ###############################
    logger.info("Loading data")
    logger.info(from_checkpoint)

    train_dataset, val_dataset = load_dataset(dataset)
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    ###################################
    ## LOADING COMPULSORY COMPONENTS ##
    ###################################
    logger.info("Preparing model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    if config_optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    elif config_optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr,
            weight_decay=weight_decay
        )
    elif config_optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr,
            weight_decay=weight_decay
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    logger.info("Start training")
    # from_checkpoint = False
    if from_checkpoint:
        checkpoint = torch.load(f"{log_folder}/checkpoint.pth.tar", map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_acc1 = checkpoint["best_acc1"]
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    if not os.path.isdir(f"{log_folder}/variance"):
        os.mkdir(f"{log_folder}/variance")
    if not os.path.isdir(f"{log_folder}/mean"):
        os.mkdir(f"{log_folder}/mean")
    for i in range(start_epoch, num_epoch):
        train_loss, train_acc1, train_acc5 = train_epoch(model, train_loader, optimizer, loss_fn, device, log_file_name, log_folder, i, clamp_value=clamp_value)

        loss, acc1, acc5 = validate_epoch(model, val_loader, loss_fn, device, log_file_name, i)

        wandb.log({
            "epoch": i,
            "train_loss": train_loss,
            "train_acc1": train_acc1,
            "train_acc5": train_acc5,
            "valid_loss": loss,
            "acc1": acc1,
            "acc5": acc5
        })

        scheduler.step() 

        is_best = acc1 > best_acc1
        
        if is_best is False:
            patient +=1
            if patient > max_patient:
                print("-----OUT OF PATIENCE-----")
                break
        else:
            patient = 0
            best_acc1 = acc1
            train_best_acc1 = train_acc1
            best_loss = loss
            train_best_loss = train_loss
            best_acc5 = acc5
            train_best_acc5 = train_acc5

        save_checkpoint(
            {
                'epoch': i + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            },
            is_best,
            log_folder
        )
    print("------END OF TRAINING------")
    print(f"Train data: loss {train_best_loss}, acc1 {train_best_acc1}, acc5 {train_best_acc5}")
    print(f"Valid data: loss {best_loss}, acc1 {best_acc1}, acc5 {best_acc5}")


def train_epoch(model, train_loader, optimizer, loss_fn, device, log_file, log_folder, epoch, clamp_value=-1):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter('Loss', ':.4e')
    weight_norm = AverageMeter('Weight Norm', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, weight_norm, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
        log_file_name=log_file)

    model.train()
    end = time.time()

    for i, (image, target) in enumerate(train_loader):
        ########################
        ## LOADING DATA TIME ###
        ########################
        data_time.update(time.time() - end)

        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(image)

        loss = loss_fn(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        weight_norm_value = cal_weight_norm(model, norm='max')
        losses.update(loss.item(), image.size(0))
        weight_norm.update(weight_norm_value, 1)
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if clamp_value != -1:
            with torch.no_grad():
                clamp_batch_norm(model, clamp_value)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.display(i + 1)

        if (i+1) % 30 == 0 and epoch < 4:
            log_var_mean(model, log_folder, epoch, i)
        elif (i+1) % 100 == 0:
            log_var_mean(model, log_folder, epoch, i)
    return losses.avg, top1.avg, top5.avg


def validate_epoch(model, valid_loader, loss_fn, device, log_file, epoch):
    
    def run_validate(valid_loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (image, target) in enumerate(valid_loader):
                i = base_progress + i
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(image)
                loss = loss_fn(output, target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), image.size(0))
                top1.update(acc1[0], image.size(0))
                top5.update(acc5[0], image.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 100 == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(valid_loader), 
        [batch_time, losses, top1, top5],
        prefix='Test: ',
        log_file_name=log_file)

    model.eval()
    run_validate(valid_loader)
    progress.display_summary()

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, log_folder, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(log_folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(log_folder, filename), os.path.join(log_folder, 'model_best.pth.tar'))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_file_name=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_file = log_file_name

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        with open(self.log_file, "a+") as f:
            f.write("\t".join(entries) + "\n")
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))
        with open(self.log_file, "a+") as f:
            f.write(" ".join(entries) + "\n")
        wandb.log

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cal_weight_norm(model, norm='max'):
    def process_layer(layer, norm):
        cum_prod = 1.0
        children = list(layer.children())
        if len(children) > 0:
            for each in children:
                cum_prod += process_layer(each, norm)
        elif hasattr(layer, "weight"):
            if isinstance(norm, int):
                cum_prod += np.log10(np.linalg.norm(layer.weight.detach().cpu().numpy().reshape(-1), ord=norm))
            else:
                cum_prod += np.log10(np.linalg.norm(layer.weight.detach().cpu().numpy().reshape(-1), ord=np.inf))
        return cum_prod
    cum_prod = process_layer(model, norm)
    return cum_prod


def log_var_mean(model, log_folder, epoch, batch):
    variance = []
    mean = []

    def process_layer(layer):
        if isinstance(layer, ModifiedConv2d) or isinstance(layer, ModifiedLinear) or isinstance(layer, ModifiedAdaptiveAvgPool2d) or isinstance(layer, ModifiedMaxPool2d):
            variance.append(layer.running_var.detach().cpu())
            mean.append(layer.running_mean.detach().cpu())
        elif len(list(layer.children())) > 0:
            for each in layer.children():
                process_layer(each)

    process_layer(model)
    torch.save(variance, f"{log_folder}/variance/variance_{epoch}_{batch}.pth")
    torch.save(mean, f"{log_folder}/mean/mean_{epoch}_{batch}.pth")


def clamp_batch_norm(model, clamp_value):
    def process_layer(layer, clamp_value_):
        if isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
            layer.weight.clamp_(min=-clamp_value_, max=clamp_value_)
        elif len(list(layer.children())) > 0:
            for child in layer.children():
                process_layer(child, clamp_value_)

    process_layer(model, clamp_value)
