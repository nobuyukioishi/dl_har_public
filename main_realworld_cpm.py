##################################################
# Main script in order to execute HAR experiments
##################################################
# Author: Marius Bock
# Email: marius.bock@uni-siegen.de
# Author: Lloyd Pellatt
# Email: lp349@sussex.ac.uk
##################################################
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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

import torch

from wimusim.dataset_configs.realworld import consts as rw_consts
from wimusim.dataset_cpm import CPM
import pickle

import numpy as np

# SEEDS = [1, 2, 3]
SEEDS = [1, 3, 4]
WANDB_ENTITY = "nobuyuki"

N_CLASSES = {
    "opportunity": 18,
    "pamap2": 12,
    "skoda": 11,
    "hhar": 7,
    "rwhar": 8,
    "shlpreview": 9,
    "realdisp": 34,
    "realdisp_wimusim_aug": 34,
    "realdisp_trad_aug": 34,
    "realdisp_ideal_n_sub": 34,
    "realdisp_ideal": 34,
    "realdisp_self": 34,
    "realworld_cpm": 8,
    "realworld_cpm": 8,
}
N_CHANNELS = {
    "opportunity": 113,
    "pamap2": 52,
    "skoda": 60,
    "hhar": 3,
    "rwhar": 3,
    "shlpreview": 22,
    "realdisp": 12,
    "realdisp_wimusim_aug": 12,
    "realdisp_trad_aug": 12,
    "realdisp_ideal_n_sub": 12,
    "realdisp_ideal": 12,
    "realdisp_self": 12,
    "realworld_cpm": 21,  # 3 (Acc) * 7 (IMUs)
    "realworld_cpm": 21,  # 3 (Acc) * 7 (IMUs)
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate an HAR model on given dataset."
    )

    parser.add_argument(
        "-d", "--dataset", type=str, help="Target dataset. Required.", required=True
    )
    parser.add_argument(
        "-v",
        "--valid_type",
        type=str,
        help="Validation type. Default split.",
        default="split",
        required=False,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model architecture. Must be the exact name of a model in the models directory."
        "Default DeepConvLSTM.",
        default="DeepConvLSTM",
    )
    parser.add_argument(
        "-e",
        "--n_epochs",
        type=int,
        help="Number of epochs to train. Default 300.",
        default=300,
        required=False,
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        help="Optimizer. Default adam.",
        default="adam",
        required=False,
    )
    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        help="Loss calculation. Default cross-entropy.",
        default="cross-entropy",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--smoothing",
        type=float,
        help="Label smoothing. Default 0.0.",
        default=0.0,
        required=False,
    )
    parser.add_argument(
        "-w",
        "--weights_init",
        type=str,
        help="Weight initialization. Default orthogonal.",
        default="orthogonal",
        required=False,
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        help="Weight decay. Default 0.0.",
        default=0.0,
        required=False,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        help="Initial learning rate. Default 1e-3.",
        default=1e-3,
        required=False,
    )
    parser.add_argument(
        "-ls",
        "--learning_rate_schedule",
        type=str,
        help="Type of learning rate schedule. Default step.",
        default="step",
        required=False,
    )
    parser.add_argument(
        "-lss",
        "--learning_rate_schedule_step",
        type=int,
        help="Initial learning rate schedule step size. If 0, learning rate schedule not applied. Default 10.",
        default=10,
        required=False,
    )
    parser.add_argument(
        "-lsd",
        "--learning_rate_schedule_decay",
        type=float,
        help="Learning rate schedule decay. Default 0.9.",
        default=0.9,
        required=False,
    )
    parser.add_argument(
        "-ws",
        "--window_size",
        type=int,
        help="Sliding window size. Default 24.",
        default=24,
        required=False,
    )
    parser.add_argument(
        "-wstr",
        "--window_step_train",
        type=int,
        help="Sliding window step size train. Default 12.",
        default=12,
        required=False,
    )
    parser.add_argument(
        "-wste",
        "--window_step_test",
        type=int,
        help="Sliding window step size test. Default 1.",
        default=1,
        required=False,
    )
    parser.add_argument(
        "-bstr",
        "--batch_size_train",
        type=int,
        help="Batch size training. Default 256.",
        default=256,
        required=False,
    )
    parser.add_argument(
        "-bste",
        "--batch_size_test",
        type=int,
        help="Batch size testing. Default 256.",
        default=256,
        required=False,
    )
    parser.add_argument(
        "-pf",
        "--print_freq",
        type=int,
        help="Print frequency (batches). Default 256.",
        default=100,
        required=False,
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Flag indicating to log results to wandb.",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--logging",
        action="store_true",
        help="Flag indicating to log results locally.",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Flag indicating to save results.",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="Flag indicating to use unweighted loss.",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="Flag indicating to use save model checkpoints.",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--lazy_load",
        action="store_true",
        help="Flag indicating to use lazy_load when loading dataset",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--scaling",
        type=str,
        help="Scaling method to apply to data. Default standardize.",
        default="standardize",
        required=False,
    )
    parser.add_argument(
        "--keep-scaling-params",
        action="store_true",
        help="Flag indicating to keep scaling parameters.",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--train_prefix",
        type=str,
        help="Prefix for training data. Default train.",
        default="train",
        required=False,
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        help="Wandb project name.",
        default="wandb-project",
        required=False,
    )

    parser.add_argument(
        "--aug_list",
        nargs="+",
        help="List of augmentation functions to apply to the data. Default None",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--aug_prob",
        nargs="+",
        help="List of probabilities for each augmentation function. Default None",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--aug_params",
        type=str,
        required=False,
        default=None,
        help="Pass a list of dictionaries as a JSON string.",
    )

    parser.add_argument(
        "--use-sim",
        action="store_true",
        help="Whether to use simulated data or not.",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--sim_aug_list",
        nargs="+",
        help="List of augmentation functions to apply to the data. Default None",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--sim_aug_prob",
        nargs="+",
        help="Probabilities for each augmentation function. Default None (0.5 for all)",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--sim_aug_params",
        type=str,
        required=False,
        default=None,
        help="Pass a list of dictionaries as a JSON string.",
    )
    parser.add_argument(
        "--paramix",
        action="store_true",
        help="Whether to use paramix or not.",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help="Print frequency (batches). Default 256.",
        default=256,
        required=False,
    )
    parser.add_argument(
        "--sim_first",
        action="store_true",
        help="Whether to use virtual IMU first for trainig.",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Flag indicating to use early stopping.",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--use_default_params",
        action="store_true",
        help="Flag indicating to use default params for WIMUSim.",
        default=False,
        required=False,
    )
    # "aug_list": [jitter, permutation],
    # "aug_prob": [0, 1],
    # "aug_params": [{"sigma": 0.2}, {"max_segments": 5, "seg_mode": "equal"}],
    args = parser.parse_args()

    return args


args = get_args()
print(paint(f"Applied Settings: "))
print(json.dumps(vars(args), indent=2, default=str))

module = import_module(f"dl_har_model.models.{args.model}")
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

sim_config = {
    "dataset": args.dataset,
    "window": args.window_size,
    "stride": args.window_step_train,
    "verbose": False,
    "resample_factor": 1,
    "n_samples": args.n_samples,
    "sim_first": args.sim_first,
    "use_sim": args.use_sim,
}


print(sim_config)
print(args.dataset == "realworld_cpm" and args.use_sim)
if args.dataset == "realworld_cpm" and args.use_sim:
    print("Use subject 1 to 13 for WIMUSim CPM")
    # sim_config["subject_ids"] = list(range(1, 14))
    # sim_config["subject_ids"] = list(range(1, 14))

    # Initialize the WIMUSimDataset (This needs to be dataset independent)
    realworld_path = "/mnt/93f1818a-0250-400e-a86c-210a9a06e78b/realworld2016_dataset/"
    if args.use_default_params:
        opt_files_dir = os.path.join(
            realworld_path, "processed/wimusim_params/opt_default"
        )
    else:
        opt_files_dir = os.path.join(realworld_path, "processed/wimusim_params/opt")
    print("Loading WIMUSim params from: ", opt_files_dir)
    
    with open(os.path.join(opt_files_dir, "opt_files_dict.pkl"), "rb") as f:
        opt_files_dict = pickle.load(f)

    act_id_dict = {
        act_name: act_idx for act_idx, act_name in enumerate(rw_consts.ACTIVITY_LIST)
    }

    B_list, P_list, D_list, H_list = [], [], [], []
    groups = []  # Used to control the sampling process
    activity_name = []
    target_list = []

    stop_id = int(config_dataset["prefix"].split("-")[1])
    print(f"Stop ID: {stop_id}")
    for sbj_id_str, sbj_opt_dict in opt_files_dict.items():
        for act_name, filenames in sbj_opt_dict.items():
            print(f"{sbj_id_str} - {act_name}: {len(filenames)} files")
            for filename in filenames:
                with open(os.path.join(opt_files_dir, filename), "rb") as f:
                    opt = pickle.load(f)
                    B_list.append(opt.env.B)
                    P_list.append(opt.env.P)
                    H_list.append(opt.env.H)

                    D_list.append(opt.env.D)
                    activity_name.append(
                        act_name
                    )  # activity_name corresponds to the dynamics parameter
                    groups.append(act_id_dict[act_name])
                    target_list.append(
                        torch.full(
                            (opt.env.D.n_samples,),
                            act_id_dict[act_name],
                            dtype=torch.int64,
                        )
                    )

        if sbj_id_str == f"p{stop_id:03d}":
            print(f"Finish loading WIMUSim params. - stop after loading p{stop_id:03d}")
            break

    train_sim_data = CPM(
        B_list=B_list,
        D_list=D_list,
        P_list=P_list,
        H_list=H_list,
        groups=groups,
        target_list=target_list,
        window=30,  # 1 seconds
        stride=15,  # 0.5 seconds
        acc_only=True,
    )  # use activity_name for the
    train_sim_data.generate_data(
        n_combinations=100  # Just for initialization. Can be any number.
    )  # This should be runned first before passed to the dataloader.


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
    "paramix": args.paramix,
    "early_stopping": args.early_stopping,
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
    wandb_logging=args.wandb,
)

# parameters used to calculate runtime
log_date = time.strftime("%Y%m%d")
log_timestamp = time.strftime("%H%M%S")

if args.wandb:
    WANDB_PROJECT = args.wandb_project_name
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config={
            "model": args.model,
            "use_sim": args.use_sim,
            **config_dataset,
            **train_args,
            **sim_config,
        },
    )

model = Model(
    N_CHANNELS[args.dataset],
    N_CLASSES[args.dataset],
    args.dataset,
    f"/{log_date}/{log_timestamp}",
).cuda()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params)


# saves logs to a file (standard output redirected)
if args.logging:
    sys.stdout = Logger(os.path.join(model.path_logs, "log"))
print(model)

if args.valid_type == "split":
    train_results, test_results, preds = split_validate(
        model,
        train_args,
        config_dataset,
        sim_data=train_sim_data if args.use_sim else None,
        seeds=SEEDS,
        verbose=True,
        keep_scaling_params=args.keep_scaling_params,
        sim_config=sim_config,
        use_sim=args.use_sim,
    )
elif args.valid_type == "loso":
    train_results, test_results, preds = loso_cross_validate(
        model, train_args, config_dataset, seeds=SEEDS, verbose=True
    )
else:
    raise ValueError(f"Invalid validation type {args.valid_type}")

run_train_analysis(train_results)
run_test_analysis(test_results)

if args.wandb:
    wandb_logging(train_results, test_results, {**config_dataset, **train_args})

if args.save_results:
    train_results.to_csv(
        os.path.join(model.path_logs, "train_results.csv"), index=False
    )
    if test_results is not None:
        test_results.to_csv(
            os.path.join(model.path_logs, "test_results.csv"), index=False
        )
    preds.to_csv(os.path.join(model.path_logs, "preds.csv"), index=False)
