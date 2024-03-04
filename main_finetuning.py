##################################################
# Main script in order to execute HAR experiments
##################################################
# Author: Marius Bock
# Email: marius.bock@uni-siegen.de
# Author: Lloyd Pellatt
# Email: lp349@sussex.ac.uk
##################################################

import argparse
import json
import os
import sys
import time
import torch
import pathlib

import numpy as np
import wandb

from dl_har_model.eval import eval_one_epoch
from dl_har_model.train import train_one_epoch

from utils import Logger, wandb_logging, paint
from importlib import import_module
from dl_har_dataloader.datasets import SensorDataset
from torch.utils.data import DataLoader

WANDB_ENTITY = 'nobuyuki'

N_CLASSES = {
             'realdisp_wimusim_aug': 34,
             'realdisp_trad_aug': 34,
             'realdisp_ideal_n_sub': 34,
             }

N_CHANNELS = {
              'realdisp_wimusim_aug': 12,
              'realdisp_trad_aug': 12,
              'realdisp_ideal_n_sub': 12,
              }


def get_args():
    parser = argparse.ArgumentParser(description='Train and evaluate an HAR model on given dataset.')

    parser.add_argument(
        '-d', '--dataset', type=str, help='Target dataset. Required.', required=True)
    parser.add_argument(
        '-v', '--valid_type', type=str, help='Validation type. Default split.', default='split', required=False)
    parser.add_argument(
        '-m', '--model', type=str, help='Model architecture. Must be the exact name of a model in the models directory.'
                                        'Default DeepConvLSTM.', default='DeepConvLSTM')
    parser.add_argument(
        '-e', '--n_epochs', type=int, help='Number of epochs to train. Default 300.', default=300, required=False)
    parser.add_argument(
        '-o', '--optimizer', type=str, help='Optimizer. Default adam.', default='adam',
        required=False)
    parser.add_argument(
        '-l', '--loss', type=str, help='Loss calculation. Default cross-entropy.', default='cross-entropy',
        required=False)
    parser.add_argument(
        '-s', '--smoothing', type=float, help='Label smoothing. Default 0.0.', default=0.0, required=False)
    parser.add_argument(
        '-w', '--weights_init', type=str, help='Weight initialization. Default orthogonal.', default='orthogonal',
        required=False)
    parser.add_argument(
        '-wd', '--weight_decay', type=float, help='Weight decay. Default 0.0.', default=0.0,
        required=False)
    parser.add_argument(
        '-lr', '--learning_rate', type=float, help='Initial learning rate. Default 1e-3.', default=1e-3, required=False)
    parser.add_argument(
        '-ls', '--learning_rate_schedule', type=str, help='Type of learning rate schedule. Default step.',
        default='step', required=False)
    parser.add_argument(
        '-lss', '--learning_rate_schedule_step', type=int,
        help='Initial learning rate schedule step size. If 0, learning rate schedule not applied. Default 10.',
        default=10, required=False)
    parser.add_argument(
        '-lsd', '--learning_rate_schedule_decay', type=float, help='Learning rate schedule decay. Default 0.9.',
        default=0.9, required=False)
    parser.add_argument(
        '-ws', '--window_size', type=int, help='Sliding window size. Default 24.',
        default=24, required=False)
    parser.add_argument(
        '-wstr', '--window_step_train', type=int, help='Sliding window step size train. Default 12.',
        default=12, required=False)
    parser.add_argument(
        '-wste', '--window_step_test', type=int, help='Sliding window step size test. Default 1.',
        default=1, required=False)
    parser.add_argument(
        '-bstr', '--batch_size_train', type=int, help='Batch size training. Default 256.',
        default=256, required=False)
    parser.add_argument(
        '-bste', '--batch_size_test', type=int, help='Batch size testing. Default 256.',
        default=256, required=False)
    parser.add_argument(
        '-pf', '--print_freq', type=int, help='Print frequency (batches). Default 256.',
        default=100, required=False)
    parser.add_argument(
        '--wandb', action='store_true', help='Flag indicating to log results to wandb.',
        default=False, required=False)
    parser.add_argument(
        '--logging', action='store_true', help='Flag indicating to log results locally.',
        default=False, required=False)
    parser.add_argument(
        '--save_results', action='store_true', help='Flag indicating to save results.',
        default=False, required=False)
    parser.add_argument(
        '--unweighted', action='store_true', help='Flag indicating to use unweighted loss.',
        default=False, required=False)
    parser.add_argument(
        '--save_checkpoints', action='store_true', help='Flag indicating to use save model checkpoints.',
        default=False, required=False)
    parser.add_argument(
        '--lazy_load', action='store_true', help='Flag indicating to use lazy_load when loading dataset',
        default=False, required=False)
    parser.add_argument(
        '--scaling', type=str, help='Scaling method to apply to data. Default standardize.',
        default='standardize', required=False)
    parser.add_argument(
        '--keep-scaling-params', action='store_true', help='Flag indicating to keep scaling parameters.',
        default=False, required=False)
    parser.add_argument('--train_prefix', type=str, help='Prefix for training data. Default train.',
                        default='train', required=False)
    parser.add_argument('--wandb-project-name', type=str, help='Wandb project name.', default='wandb-project', required=False)
    parser.add_argument("--checkpoint-path", type=str, help="Path to a checkpoint to load.", default=None, required=False)
    parser.add_argument("--subject-id", type=int, help="Subject ID to fine-tune.", default=None, required=True)
    parser.add_argument("--seed", type=int, help="Random seed.", default=1, required=False)
    args = parser.parse_args()

    return args


args = get_args()
print(paint(f"Applied Settings: "))
print(json.dumps(vars(args), indent=2, default=str))

module = import_module(f'dl_har_model.models.{args.model}')
print(paint(f"Applied Model: "))
Model = getattr(module, args.model)



config_dataset = {
    "dataset": args.dataset,
    "window": args.window_size,
    "stride": args.window_step_train,
    "stride_test": args.window_step_test,
    "path_processed": f"data/{args.dataset}",
    "lazy_load": args.lazy_load,
    "scaling": args.scaling,
    "mean": np.array([-2.43506481, 2.67243986, -1.90483712, 0.03534253, 0.00532397, 0.00616601, -3.99648382, -1.93164316, -0.05685239, -0.03086638,  0.01071271, 0.01518551]), # To be loaded from the train dataset
    "std": np.array([6.98216677, 6.29808711, 6.10557477, 1.16950008, 1.18748655, 1.30134347, 6.64125895, 6.44705467, 5.98579538, 1.22379754, 1.14913085, 1.2805076]), # To be loaded from the train dataset
}

train_args = {
    "batch_size_train": args.batch_size_train,
    "batch_size_test": args.batch_size_test,
    "optimizer": args.optimizer,
    "use_weights": args.unweighted,
    "lr": args.learning_rate,
    "lr_schedule": args.learning_rate_schedule,
    "lr_step": args.learning_rate_schedule_step,
    "lr_decay": args.learning_rate_schedule_decay,
    "weights_init": args.weights_init,
    "epochs": args.n_epochs,
    "print_freq": args.print_freq,
    "loss": args.loss,
    "smoothing": args.smoothing,
    "weight_decay": args.weight_decay,
    "save_checkpoints": args.save_checkpoints,
    "subject_id": args.subject_id
}

config = dict(
    seeds=[args.seed],
    model=args.model,
    valid_type=args.valid_type,
    batch_size_train=args.batch_size_train,
    epochs=args.n_epochs,
    optimizer=args.optimizer,
    loss=args.loss,
    smoothing=args.smoothing,
    use_weights=args.unweighted,
    lr=args.learning_rate,
    lr_schedule=args.learning_rate_schedule,
    lr_step=args.learning_rate_schedule_step,
    lr_decay=args.learning_rate_schedule_decay,
    weights_init=args.weights_init,
    weight_decay=args.weight_decay,
    batch_size_test=args.batch_size_test,
    wandb_logging=args.wandb
)

# parameters used to calculate runtime
log_date = time.strftime('%Y%m%d')
log_timestamp = time.strftime('%H%M%S')

# Load the model from the checkopoint
checkpoint_path = pathlib.Path(args.checkpoint_path)
assert checkpoint_path.exists(), f"Checkpoint path {checkpoint_path} does not exist."
model = Model(N_CHANNELS[args.dataset], N_CLASSES[args.dataset], args.dataset, f"/{log_date}/{log_timestamp}").cuda()
model.path_checkpoints = checkpoint_path
path_checkpoint = model.path_checkpoints / "checkpoint_best.pth"

# check if checkpoint exists
if not path_checkpoint.exists():
    raise ValueError(f"Checkpoint {path_checkpoint} does not exist")

checkpoint = torch.load(path_checkpoint)
model.load_state_dict(checkpoint["model_state_dict"])

# saves logs to a file (standard output redirected)
if args.logging:
    sys.stdout = Logger(os.path.join(model.path_logs, 'log'))
print(model)

subject_id = args.subject_id
batch_size = args.batch_size_train
print(f"train prefix: train-{subject_id}-ft")
print(f"val prefix: val-{subject_id}")

train_ft_dataset = SensorDataset(prefix=f"train-{subject_id}-ft", **config_dataset)
if subject_id in [14, 15, 16, 17]:
    val_dataset = SensorDataset(prefix=f'val-{subject_id}', **config_dataset)
elif subject_id in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14]:
    val_dataset = SensorDataset(prefix=f'test-{subject_id}', **config_dataset)
else:
    raise ValueError(f"Invalid subject ID: {subject_id}")

loader_train_ft = DataLoader(train_ft_dataset, batch_size, True, pin_memory=True, worker_init_fn=np.random.seed(int(args.seed)))
loader_val = DataLoader(val_dataset, batch_size, False, pin_memory=True, worker_init_fn=np.random.seed(int(args.seed)))

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


criterion = torch.nn.CrossEntropyLoss()

freeze = False
# Freeze the layers except for the classifier
if freeze:
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

print("Check the initial results on validation set")
loss_ft, acc_ft, fm_ft, fw_ft = eval_one_epoch(model, loader_train_ft, criterion)
print(f"\tTrain: loss: {loss_ft:.3f}, acc: {acc_ft:.3f}, fm: {fm_ft:.3f}, fw: {fw_ft:.3f}")
loss_val, acc_val, fm_val, fw_val = eval_one_epoch(model, loader_val, criterion)
print(f"\t  Val: loss: {loss_val:.3f}, acc: {acc_val:.3f}, fm: {fm_val:.3f}, fw: {fw_val:.3f}")


if args.wandb:
    WANDB_PROJECT = args.wandb_project_name
    initial_results = {
        "train_loss_init": loss_ft,
        "train_acc_init": acc_ft,
        "train_fm_init": fm_ft,
        "train_fw_init": fw_ft,
        "val_loss_init": loss_val,
        "val_acc_init": acc_val,
        "val_fm_init": fm_val,
        "val_fw_init": fw_val
    }
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
               config={"model": args.model, "seed": args.seed, **config_dataset, **train_args, **initial_results})


print_freq = 100
centerloss = False
lr_cent = 1e-4
beta = 0.5,
mixup = False,
alpha = 0.5,
verbose = False
for i in range(train_args["epochs"]):
    print(f"Epoch {i+1}")
    train_one_epoch(model, loader_train_ft, criterion, optimizer, print_freq, centerloss, lr_cent, beta, mixup, alpha, verbose)
    loss_ft, acc_ft, fm_ft, fw_ft = eval_one_epoch(model, loader_train_ft, criterion)
    print(f"\tTrain: loss: {loss_ft:.3f}, acc: {acc_ft:.3f}, fm: {fm_ft:.3f}, fw: {fw_ft:.3f}")
    loss_val, acc_val, fm_val, fw_val = eval_one_epoch(model, loader_val, criterion)
    print(f"\t  Val: loss: {loss_val:.3f}, acc: {acc_val:.3f}, fm: {fm_val:.3f}, fw: {fw_val:.3f}")
    if args.wandb:
        wandb.log({"train_loss": loss_ft,
                   "train_acc": acc_ft,
                   "train_fm": fm_ft,
                   "train_fw": fw_ft,
                   "val_loss": loss_val,
                   "val_acc": acc_val,
                   "val_fm": fm_val,
                   "val_fw": fw_val})
