python main.py -d realworld -v split --window_size 30 --window_step_train 15 --n_epochs 100 --lazy_load --batch_size_train 256 --keep-scaling-params --model DeepConvLSTM --train_prefix train --wandb --wandb-project-name realworld
python main.py -d realworld -v split --window_size 30 --window_step_train 15 --n_epochs 100 --lazy_load --batch_size_train 512 --keep-scaling-params --model DeepConvLSTM --train_prefix train --wandb --wandb-project-name realworld
python main.py -d realworld -v split --window_size 30 --window_step_train 15 --n_epochs 100 --lazy_load --batch_size_train 1024 --keep-scaling-params --model DeepConvLSTM --train_prefix train --wandb --wandb-project-name realworld
