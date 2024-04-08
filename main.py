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

import wandb

from dl_har_analysis.analysis import run_train_analysis, run_test_analysis

from dl_har_model.train import split_validate, loso_cross_validate
from utils import Logger, wandb_logging, paint
from importlib import import_module

from dl_har_dataloader.augmentation import jitter, scaling, permutation, rotate, time_warp, magnitude_warp


SEEDS = [1, 2, 3]
WANDB_ENTITY = 'nobuyuki'

N_CLASSES = {'opportunity': 18,
             'pamap2': 12,
             'skoda': 11,
             'hhar': 7,
             'rwhar': 8,
             'shlpreview': 9,
             'realdisp': 34,
             'realdisp_wimusim_aug': 34,
             'realdisp_trad_aug': 34,
             'realdisp_ideal_n_sub': 34,
             }
N_CHANNELS = {'opportunity': 113,
              'pamap2': 52,
              'skoda': 60,
              'hhar': 3,
              'rwhar': 3,
              'shlpreview': 22,
              'realdisp': 12,
              'realdisp_wimusim_aug': 12,
              'realdisp_trad_aug': 12,
              'realdisp_ideal_n_sub': 12,
              }

aug_dict = {
    "jitter": jitter,
    "scaling": scaling,
    "permutation": permutation,
    "rotate": rotate,
    "time_warp": time_warp,
    "magnitude_warp": magnitude_warp
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
    parser.add_argument("--aug_list", nargs='+', help="List of augmentation functions to apply to the data. Default None", default=None, required=False)
    parser.add_argument("--aug_prob", nargs='+', help="List of probabilities for each augmentation function. Default None", default=None, required=False)
    parser.add_argument("--aug_params", type=json.loads, help="List of parameter dictionaries for each augmentation. Default None", default=None, required=False)

    # "aug_list": [jitter, permutation],
    # "aug_prob": [0, 1],
    # "aug_params": [{"sigma": 0.2}, {"max_segments": 5, "seg_mode": "equal"}],
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
    "prefix": args.train_prefix,
}


# Augmentation settings
if args.aug_list is not None:
    config_dataset["aug_list"] = [aug_dict[key] for key in args.aug_list]
    if args.aug_prob is not None:
        config_dataset["aug_prob"] = [float(prob) for prob in args.aug_prob]
    else:
        config_dataset["aug_prob"] = [0.0 for _ in range(len(args.aug_list))]

    if args.aug_params is not None:
        config_dataset["aug_params"] = args.aug_params
    else:
        config_dataset["aug_params"] = [None for _ in range(len(args.aug_list))]

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
    "save_checkpoints": args.save_checkpoints
}

config = dict(
    seeds=SEEDS,
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

if args.wandb:
    WANDB_PROJECT = args.wandb_project_name
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config={"model": args.model, **config_dataset, **train_args})

model = Model(N_CHANNELS[args.dataset], N_CLASSES[args.dataset], args.dataset, f"/{log_date}/{log_timestamp}").cuda()

# saves logs to a file (standard output redirected)
if args.logging:
    sys.stdout = Logger(os.path.join(model.path_logs, 'log'))
print(model)

if args.valid_type == 'split':
    train_results, test_results, preds = \
        split_validate(model, train_args, config_dataset, seeds=SEEDS, verbose=True, keep_scaling_params=args.keep_scaling_params)
elif args.valid_type == 'loso':
    train_results, test_results, preds = \
        loso_cross_validate(model, train_args, config_dataset, seeds=SEEDS, verbose=True)
else:
    raise ValueError(f"Invalid validation type {args.valid_type}")

run_train_analysis(train_results)
run_test_analysis(test_results)

if args.wandb:
    wandb_logging(train_results, test_results, {**config_dataset, **train_args})

if args.save_results:
    train_results.to_csv(os.path.join(model.path_logs, 'train_results.csv'), index=False)
    if test_results is not None:
        test_results.to_csv(os.path.join(model.path_logs, 'test_results.csv'), index=False)
    preds.to_csv(os.path.join(model.path_logs, 'preds.csv'), index=False)
