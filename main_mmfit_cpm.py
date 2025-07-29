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

# from wimusim.dataset_configs.realworld import consts as rw_consts
from wimusim.dataset_configs.mmfit import consts as mmfit_consts
from wimusim.dataset_cpm import CPM
import pickle

import numpy as np

SEEDS = [1, 2, 3]
# SEEDS = [1, 3, 4]
# SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# SEEDS = [2, 7, 9]
WANDB_ENTITY = "nobuyuki"

N_CLASSES = {
    "mmfit": 11,
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
}
N_CHANNELS = {
    "mmfit": 18,
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
    # parser.add_argument(
    #     "--train_prefix",
    #     type=str,
    #     help="Prefix for training data. Default train.",
    #     default="train",
    #     required=False,
    # )
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
    parser.add_argument(
        "--n_sub",
        type=int,
        help="Number of subjects for training.",
        default=6,
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

real_train_files = ["train-1", "train-2", "train-3", "train-4", "train-5", "train-6"]
train_prefix = real_train_files[: args.n_sub]
print(f"Train prefix: {train_prefix} (real training data)")

config_dataset = {
    "dataset": args.dataset,
    "window": args.window_size,
    "stride": args.window_step_train,
    "stride_test": args.window_step_test,
    "path_processed": f"data/{args.dataset}",
    "lazy_load": args.lazy_load,
    "scaling": args.scaling,
    "prefix": train_prefix,
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
    "use_default_params": args.use_default_params,
}

print(sim_config)
print(args.dataset == "mmfit" and args.use_sim)
if args.dataset == "mmfit" and args.use_sim:

    # sim_config["subject_ids"] = list(range(1, 14))
    # sim_config["subject_ids"] = list(range(1, 14))
    subject_session_id_dict = {
        0: "00",
        1: "03",
        2: "07",
        3: "12",
        4: "13",
        5: "16"
    }
    # Initialize the WIMUSimDataset (This needs to be dataset independent)
    # I need to set the opt_files_dir
    mmfit_path = "./data/mmfit/"
    wimusim_params_dir = mmfit_path + "wimusim_params/"
    print("Loading WIMUSim params from: ", wimusim_params_dir)

    act_id_dict = {
        act_name: act_idx for act_idx, act_name in enumerate(mmfit_consts.ACTIVITY_LIST)
    }

    B_list, P_list, D_list, H_list = [], [], [], []
    activity_name = []
    target_list = []

    print("Loading WIMUSim params...", args.n_sub)
    for subject_id in range(args.n_sub):
        print(f"Loading WIMUSim params for subject {subject_id + 1}... use_default_params={args.use_default_params}")
        if args.use_default_params:
            wimusim_params_filepath = f"{mmfit_path}/wimusim_params/w{subject_session_id_dict[subject_id]}_wimusim_params_default_dict.pkl"
        else:
            wimusim_params_filepath = f"{mmfit_path}/wimusim_params/w{subject_session_id_dict[subject_id]}_wimusim_params_dict.pkl"

        with open(wimusim_params_filepath, "rb") as f:
            # print(wimusim_params_dict["B"])
            # print(wimusim_params_dict["D"])
            # print(wimusim_params_dict["P"])
            # print(wimusim_params_dict["H"])
            wimusim_params_dict = pickle.load(f)
            B_list.append(wimusim_params_dict["B"])
            D_list.append(wimusim_params_dict["D"])
            P_list.append(wimusim_params_dict["P"])
            H_list.append(wimusim_params_dict["H"])
            target_list.append(wimusim_params_dict["target"])

    train_sim_data = CPM(
        B_list=B_list,
        D_list=D_list,
        P_list=P_list,
        H_list=H_list,
        target_list=target_list,
        window=60,  # 2 seconds
        stride=30,  # 1.0 seconds
        acc_only=False,
    )  # use activity_name for the
    train_sim_data.generate_data(
        n_combinations=100  # Just for initialization. Can be any number.
    )  # This should be runned first before passed to the dataloader.
else:
    print("Not using WIMUSim CPM dataset.")

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
