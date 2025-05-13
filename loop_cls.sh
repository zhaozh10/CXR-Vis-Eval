#!/usr/bin/env bash



# 默认数据集和数据比例
# datasets=("nih" "rsna")
# datasets=("rsna" "nih")
datasets=("nih")
# datasets=("cheststruct")
# datasets=("rsna")
data_pcts=(0.01 0.1 1)
# data_pcts=( 0.01 0.1)
# data_pcts=(0.1 1 0.01)
# data_pcts=(0.01)
# data_pcts=(1)

# ckpt_path_list=("../dMAE/output_dir/dmae/epoch_400.pth" "../DeepMIM/output_dir/deepmim/epoch_400.pth"  "../maskalign/output_dir/maskalign/epoch_400.pth")
ckpt_path_list=("../GazeMIM/output_dir/gazemim-wTv4-depth1-head8/checkpoint-300.pth")
# ckpt_path_list=(None)
eval_type="linear"
base_model="vit_base_p16"
# 种子值数组
seed_values=(43 44 45)

# 外层循环遍历数据集
for ckpt_path in "${ckpt_path_list[@]}"
do
    for dataset in "${datasets[@]}"
    do
        # 中层循环遍历数据比例
        for data_pct in "${data_pcts[@]}"
        do
            # 内层循环遍历种子值
            for seed_value in "${seed_values[@]}"
            do
                # PYTHONPATH="$(dirname $0)/..":$PYTHONPATH  CUDA_VISIBLE_DEVICES=0 python downstream/models/mgca/mgca_classifier.py --gpus 1 --batch_size 64 --learning_rate 2e-3 --base_model $base_model  --dataset $dataset --data_pct $data_pct --seed $seed_value  --eval_type $eval_type --ckpt_path $ckpt_path
                PYTHONPATH="$(dirname $0)/..":$PYTHONPATH  CUDA_VISIBLE_DEVICES=0 python downstream/models/mgca/mgca_classifier.py --gpus 1 --batch_size 64 --learning_rate 2e-3 --base_model $base_model  --dataset $dataset --data_pct $data_pct --seed $seed_value  --eval_type $eval_type --ckpt_path $ckpt_path
                # CUDA_VISIBLE_DEVICES=0 python downstream/models/mgca/mgca_classifier.py --gpus 1  --base_model resnet_50  --dataset $dataset --data_pct $data_pct --seed $seed_value
            done
        done
    done
done



# for dataset in "${datasets[@]}"
# do
#     # 中层循环遍历数据比例
#     for data_pct in "${data_pcts[@]}"
#     do
#         # 内层循环遍历种子值
#         for seed_value in "${seed_values[@]}"
#         do
#             PYTHONPATH="$(dirname $0)/..":$PYTHONPATH  CUDA_VISIBLE_DEVICES=0 python downstream/models/mgca/mgca_classifier.py --gpus 1 --batch_size 64 --learning_rate 2e-3 --base_model $base_model  --dataset $dataset --data_pct $data_pct --seed $seed_value  --eval_type $eval_type --ckpt_path $ckpt_path
#             # PYTHONPATH="$(dirname $0)/..":$PYTHONPATH  CUDA_VISIBLE_DEVICES=0 python downstream/models/mgca/mgca_classifier.py --gpus 1 --batch_size 64 --learning_rate 2e-3 --base_model $base_model  --dataset $dataset --data_pct $data_pct --seed $seed_value  --eval_type $eval_type
#             # CUDA_VISIBLE_DEVICES=0 python downstream/models/mgca/mgca_finetuner.py --gpus 1  --base_model resnet_50  --dataset $dataset --data_pct $data_pct --seed $seed_value
#         done
#     done
# done




