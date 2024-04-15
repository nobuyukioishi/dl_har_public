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

# SEEDS = [1, 2, 3]
SEEDS = [1, 2, 3]
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
        "--use_sim",
        action="store_true",
        help="Flag indicating to use unweighted loss.",
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
    "resample_factor": 2,
}

# WIMUSim Augmentation settings
if args.sim_aug_list is not None:
    sim_config["aug_list"] = args.sim_aug_list
    if args.sim_aug_prob is not None:
        sim_config["aug_prob"] = [float(prob) for prob in args.sim_aug_prob]
    else:
        sim_config["aug_prob"] = [0.5 for _ in range(len(args.sim_aug_list))]

    if args.sim_aug_params is not None:
        sim_config["aug_params"] = json.loads(args.sim_aug_params)
    else:
        sim_config["aug_params"] = [None for _ in range(len(args.sim_aug_list))]

print(sim_config)

if args.dataset == "realdisp_ideal" and args.use_sim:
    print("Use ideal scenario (subject 1-10) for WIMUSim augmentation")
    sim_config["wimusim_params_path_dict"] = {
        1: "../WIMUSim/data/REALDISP/saved_params/ideal_sub1_wimusim_params_opt_with_labels.pt",
        2: "../WIMUSim//data/REALDISP/saved_params/ideal_sub2_wimusim_params_opt_with_labels.pt",
        3: "../WIMUSim//data/REALDISP/saved_params/ideal_sub3_wimusim_params_opt_with_labels.pt",
        4: "../WIMUSim//data/REALDISP/saved_params/ideal_sub4_wimusim_params_opt_with_labels.pt",
        5: "../WIMUSim//data/REALDISP/saved_params/ideal_sub5_wimusim_params_opt_with_labels.pt",
        6: "../WIMUSim//data/REALDISP/saved_params/ideal_sub6_wimusim_params_opt_with_labels.pt",
        7: "../WIMUSim//data/REALDISP/saved_params/ideal_sub7_wimusim_params_opt_with_labels.pt",
        8: "../WIMUSim//data/REALDISP/saved_params/ideal_sub8_wimusim_params_opt_with_labels.pt",
        9: "../WIMUSim//data/REALDISP/saved_params/ideal_sub9_wimusim_params_opt_with_labels.pt",
        10: "../WIMUSim//data/REALDISP/saved_params/ideal_sub10_wimusim_params_opt_with_labels.pt",
    }
elif args.dataset == "realdisp_self" and args.use_sim:
    print("Use ideal scenario (subject 1-17) for WIMUSim augmentation")
    sim_config["wimusim_params_path_dict"] = {
        1: "../WIMUSim/data/REALDISP/saved_params/ideal_sub1_wimusim_params_opt_with_labels.pt",
        2: "../WIMUSim//data/REALDISP/saved_params/ideal_sub2_wimusim_params_opt_with_labels.pt",
        3: "../WIMUSim//data/REALDISP/saved_params/ideal_sub3_wimusim_params_opt_with_labels.pt",
        4: "../WIMUSim//data/REALDISP/saved_params/ideal_sub4_wimusim_params_opt_with_labels.pt",
        5: "../WIMUSim//data/REALDISP/saved_params/ideal_sub5_wimusim_params_opt_with_labels.pt",
        6: "../WIMUSim//data/REALDISP/saved_params/ideal_sub6_wimusim_params_opt_with_labels.pt",
        7: "../WIMUSim//data/REALDISP/saved_params/ideal_sub7_wimusim_params_opt_with_labels.pt",
        8: "../WIMUSim//data/REALDISP/saved_params/ideal_sub8_wimusim_params_opt_with_labels.pt",
        9: "../WIMUSim//data/REALDISP/saved_params/ideal_sub9_wimusim_params_opt_with_labels.pt",
        10: "../WIMUSim//data/REALDISP/saved_params/ideal_sub10_wimusim_params_opt_with_labels.pt",
        11: "../WIMUSim/data/REALDISP/saved_params/ideal_sub11_wimusim_params_opt_with_labels.pt",
        12: "../WIMUSim//data/REALDISP/saved_params/ideal_sub12_wimusim_params_opt_with_labels.pt",
        13: "../WIMUSim//data/REALDISP/saved_params/ideal_sub13_wimusim_params_opt_with_labels.pt",
        14: "../WIMUSim//data/REALDISP/saved_params/ideal_sub14_wimusim_params_opt_with_labels.pt",
        15: "../WIMUSim//data/REALDISP/saved_params/ideal_sub15_wimusim_params_opt_with_labels.pt",
        16: "../WIMUSim//data/REALDISP/saved_params/ideal_sub16_wimusim_params_opt_with_labels.pt",
        17: "../WIMUSim//data/REALDISP/saved_params/ideal_sub17_wimusim_params_opt_with_labels.pt",
    }
else:
    if args.use_sim:
        raise ValueError(f"Invalid evaluation scenario {args.dataset}")


# Augmentation settings
if args.aug_list is not None:
    config_dataset["aug_list"] = args.aug_list
    if args.aug_prob is not None:
        config_dataset["aug_prob"] = [float(prob) for prob in args.aug_prob]
    else:
        config_dataset["aug_prob"] = [0.0 for _ in range(len(args.aug_list))]

    if args.aug_params is not None:
        config_dataset["aug_params"] = json.loads(args.aug_params)
    else:
        config_dataset["aug_params"] = [None for _ in range(len(args.aug_list))]

# This will be used to determine the rot_list in the WIMUSim augmentation
# You can ignore the assertion if you don't use the rotation augmentation
if args.dataset == "realdisp_ideal":
    scenario = "ideal"
elif args.dataset == "realdisp_self":
    scenario = "self"
else:
    scenario = None
    assert "Unexpected scenario."

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
    "scenario": scenario,
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

# saves logs to a file (standard output redirected)
if args.logging:
    sys.stdout = Logger(os.path.join(model.path_logs, "log"))
print(model)

if args.valid_type == "split":
    train_results, test_results, preds = split_validate(
        model,
        train_args,
        config_dataset,
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
