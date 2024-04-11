# Unified_Layer_Skipping

* [Overview](#overview)
* [Requirements](#requirements)
* [Quick to Use](#quick-to-use)
* [Citation](#citation)
* [Contact](#contact)


## Overview
<p align="center">
  <img src="https://github.com/Adaxry/Unified_Layer_Skipping/blob/main/figures/overview.png" alt="overview" width="600"/>
</p>
<p align="center">
  Overview comparisions of serveral related approaches.
</p>


We propose a Unified Layer Skipping strategy for Large Language Models that selects and skips computational layers based on the target speedup ratio, providing stable acceleration, preserving performance, and supporting popular acceleration techniques, thereby outperforming existing dynamic computation methods in both inference performance and actual model throughput.

Details usages and commands for SFT and decoding will be uploaded within a week.

## Quick to Use
Here, we take the Bloomz-7B as an example to show how to conduct fine-tuning and decoding with Unified Layer Skipping. 

+ Fine-Tuning

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1
export CXX=g++

export MASTER_ADDR="${CHIEF_IP:=localhost}"
MASTER_PORT=$((1 + $RANDOM % 99999))

your_transformers_path=""

export PYTHONPATH=$PYTHONPATH:"$your_transformers_path/src"
echo $PYTHONPATH

gpu_num=8
accum_num=16

train()
{
    target_speedup=$1
    model_name=bloomz_7b_block512_skip_uniform_target_speedup${target_speedup}x
    your_work_path="your_work_path"
    train_path=$your_transformers_path/examples/pytorch/language-modeling/run_clm_llms.py
    premodel=$your_work_path/bloomz-7b1-mt
    
    model_save=$your_work_path/checkpoint/$model_name
    LOG_FILE=$your_work_path/log.${model_name}
    
    #data_dir=/apdcephfs/share_47076/yijinliu/transformers/data
    export TRANSFORMERS_CACHE=${data_dir}/cache
    export HF_HOME=${data_dir}/cache/
    export TORCH_EXTENSIONS_DIR=${data_dir}/cache/torch_extension/${model_name}
    export OMP_NUM_THREADS=20
    TOKENIZERS_PARALLELISM=false
    HOST_NUM=1
    INDEX=0
    train_files=./data/*.json
    
    torchrun --nnodes $HOST_NUM --node_rank $INDEX --nproc_per_node $gpu_num \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
        ${train_path} \
        --deepspeed $your_work_path/train/deepspeed_config.json \
        --model_name_or_path ${premodel} \
        --train_file $train_files \
        --preprocessing_num_workers 16 \
        --dataloader_num_workers 16 \
        --dataloader_pin_memory True \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps $accum_num \
        --num_train_epochs 2.0 \
        --save_strategy "steps" \
        --save_steps 10000 \
        --save_total_limit 50 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --block_size 512 \
        --do_train \
        --evaluation_strategy "no" \
        --validation_split_percentage 0 \
        --fp16 True \
        --fp16_full_eval True \
        --streaming \
        --ddp_timeout 36000 \
        --seed 1 \
        --gradient_checkpointing False \
        --output_dir ${model_save} \
        --cache_dir ${data_dir}/cache/ \
        --freeze_emb True \
        --target_speedup $target_speedup \
        --overwrite_output_dir \
        --overwrite_cache \
        2>&1 |tee ${LOG_FILE}
}

(train "2") # for 2x target speedup
(train "3") # for 3x target speedup
(train "5") # for 5x target speedup
(train "10") # for 10x target speedup



```

+ Decoding



## Requirements
+ transformers>=4.28.0.dev0+
+ python>=3.8.0
+ torch>=1.10
+ deepspeed>=0.8.3+
+ datasets>=2.9.0+


## Contact
Please feel free to contact us (yijinliu@tencent.com) for any further questions.  
