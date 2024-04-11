# Unified_Layer_Skipping

* [Overview](#overview)
* [Requirements](#requirements)
* [Quick to Use](#quick-to-use)
* [Experiments](#experiments)
* [Contact](#contact)


## Overview
<p align="center">
  <img src="https://github.com/Adaxry/Unified_Layer_Skipping/blob/main/figures/overview.png" alt="overview" width="600"/>
</p>
<p align="center">
  Overview comparisions of serveral related approaches.
</p>


We propose a Unified Layer Skipping strategy for Large Language Models that selects and skips computational layers based on the target speedup ratio, providing stable acceleration, preserving performance, and supporting popular acceleration techniques (e.g., batch decoding and KV caching). Our pre-print paper is available at [here](https://arxiv.org/abs/2404.06954).


## Requirements
+ transformers>=4.28.0.dev0+
+ python>=3.8.0
+ torch>=1.10
+ deepspeed>=0.8.3+
+ datasets>=2.9.0+

## Quick to Use
Here, we take the Bloomz-7B as an example to show how to conduct fine-tuning and decoding with Unified Layer Skipping. 

+ Replace Files to Support Layer Skipping
```
your_transformers_path=""  # change to your transformers path
mv ./codes/modeling_bloom.py $your_transformers_path/src/transformers/models/bloom/modeling_bloom.py
mv ./codes/run_clm_llms.py $your_transformers_path/examples/pytorch/language-modeling/run_clm_llms.py

```

+ Fine-Tuning

```bash
#!/usr/bin/bash 
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

train()
{
    target_speedup=$1
    model_name=bloomz_7b_block512_skip_uniform_target_speedup${target_speedup}x
    your_work_path="your_work_path"
    train_path=$your_transformers_path/examples/pytorch/language-modeling/run_clm_llms.py
    premodel=$your_work_path/bloomz-7b1-mt    
    model_save=$your_work_path/checkpoint/$model_name
    LOG_FILE=$your_work_path/log.${model_name}
    export TRANSFORMERS_CACHE=${data_dir}/cache
    export HF_HOME=${data_dir}/cache/
    export TORCH_EXTENSIONS_DIR=${data_dir}/cache/torch_extension/${model_name}
    export OMP_NUM_THREADS=20
    TOKENIZERS_PARALLELISM=false
    HOST_NUM=1
    INDEX=0
    gpu_num=8
    accum_num=16
    train_files=./data/*.json # change to your json data that is compatible with the transformers framework
    
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

```bash
#!/bin/bash
target_speedup="5"  # keep this value the same as the training stage
work_dir=$your_work_path
model_name=$your_model_name
code_name="./codes/inference_skip_decode.py"
post_ins_test=""
post_ins_test="_post_ins_test"
no_repeat_ngram_size=3
log_name=log
result_dir=$work_dir/results/${model_name}

if [ ! -d $result_dir ]; then
    mkdir -p $result_dir
fi

if [ $no_repeat_ngram_size -gt 0 ]; then
  result_dir="${result_dir}_ngram${no_repeat_ngram_size}"
fi

ins_file="instruct_inf"
template="prompt_input"

if [ -n "$post_ins_test" ]; then
  template="prompt_input_above"
  ins_file=${ins_file}_above #
fi

ins_file=${ins_file}.txt 
post_ins_file=${post_ins_file}.txt 

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_CACHE=$your_work_path/data/cache
export HF_HOME=$your_work_path/data/cache
export TF_ENABLE_ONEDNN_OPTS=0

inference()
{
    src=$1
    tgt=$2
    log_name=${log_name}
    python $work_dir/test/$code_name \
        --model-name-or-path ${work_dir}/checkpoint/$model_name \
        -lp $src-$tgt \
        -t 0.1 \
        -b 4 \
        -sa 'beam' \
        --batch 1 \
        --no-repeat-ngram-size $no_repeat_ngram_size \
        -ins ./test/$ins_file \
        -i  ./test/wmt22/newstest22.$src-$tgt.$src \
        -tp $template \
        -o $result_dir/${src}-${tgt}.out \
        --target_speedup $target_speedup 

    # get bleu score
    python ./test/sacre_verbose.py $result_dir/${src}-${tgt}.out.hyp ./test/wmt22/newstest22.${src}-${tgt}.${tgt} $result_dir/bleu $tgt >> $result_dir/$log_name
}

(export CUDA_VISIBLE_DEVICES=0;inference de en ;sleep 150)& \
(export CUDA_VISIBLE_DEVICES=1;inference en de)& \
(export CUDA_VISIBLE_DEVICES=2;inference en zh ;sleep 150)& \
(export CUDA_VISIBLE_DEVICES=3;inference zh en)
wait

```


## Experiments

Datas used in our paper can be obtained at [here](https://github.com/Adaxry/Post-Instruction/blob/main/data/README.md).
The following are our results on the WMT22 test set, tested on a single Nvidia A100-80G node. Please note that the number of tokens are counted by the sum of the prefix and output tokens. It can be observed that under the same target speedup ratio, our method can achieve better translation quality and throughput.
<p align="center">
  <img src="https://github.com/Adaxry/Unified_Layer_Skipping/blob/main/figures/throughput.png" alt="overview" width="400"/>
</p>


## Contact
Please feel free to contact us (yijinliu@tencent.com) for any further questions.  
