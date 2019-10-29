#!/bin/sh

#VARIABLE DEFINITIONS
model_name=vgg
data_dir=/data/imagenet/validation
graph=models/vgg16_slim_pretrained/vgg16_slim_pretrained.pb
ckpt=models/vgg16_slim_pretrained/vgg16_slim_pretrained.ckpt

# RUN VALIDATION SCRIPT
python run_inference_dataset.py \
    --model_name $model_name \
    --data_dir $data_dir \
    --graph_path $graph \
    --ckpt_path $ckpt

# EVALUATE TOP1 AND TOP5
python eval.py outputs/$model_name 

